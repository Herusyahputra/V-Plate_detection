import math
from sympy import Symbol, Eq, solve


class CoordinateSystem(object):
    def __init__(self, cam1_3d_coordinate, cam2_3d_coordinate):
        """

        Args:
            cam1_3d_coordinate ():
            cam2_3d_coordinate ():
        """
        self.cam1_3d_coordinate = cam1_3d_coordinate
        self.cam2_3d_coordinate = cam2_3d_coordinate
        self.point1_alpha_l = None
        self.point1_beta_l = None

        self.point2_alpha_l = None
        self.point2_beta_l = None

        self.point1_alpha_r = None
        self.point1_beta_r = None

        self.point2_alpha_r = None
        self.point2_beta_r = None

        self.target_point1 = None
        self.target_point2 = None

        self.dis = None

    def set_cam1_3d_coordinate(self, cam1_3d_coordinate):
        self.cam1_3d_coordinate = cam1_3d_coordinate

    def set_cam2_3d_coordinate(self, cam2_3d_coordinate):
        self.cam2_3d_coordinate = cam2_3d_coordinate

    def calculate_dis(self):

        print("~~~~~~~~~~~~ CoordinateSystem.calculate_dis(): !!! START !!! ~~~~~~~~~~~~~~")
        if self.point1_alpha_l == 0.0 or self.point1_beta_l == 0.0:
            print("A")
            print("CoordinateSystem.calculate_dis(): !!! No point1_alpha_l or No point1_beta_l !!!")
        elif self.point1_alpha_r == 0.0 or self.point1_beta_r == 0.0:
            print("B")
            print("CoordinateSystem.calculate_dis(): !!! No point1_alpha_r or No point1_beta_r !!!")
        elif self.point2_alpha_l == 0.0 or self.point2_beta_l == 0.0:
            print("C")
            print("CoordinateSystem.calculate_dis(): !!! No point2_alpha_l or No point2_beta_l !!!")
        elif self.point2_alpha_r == 0.0 or self.point2_beta_r == 0.0:
            print("D")
            print("CoordinateSystem.calculate_dis(): !!! No point2_alpha_r or No point2_beta_r !!!")
        else:
            print("2", self.cam1_3d_coordinate)
            print("CoordinateSystem.calculate_dis(): self.point1_alpha_l=", self.point1_alpha_l)
            print("CoordinateSystem.calculate_dis(): self.point2_alpha_l=", self.point2_alpha_l)
            print("CoordinateSystem.calculate_dis(): self.point1_beta_l=", self.point1_beta_l)
            print("CoordinateSystem.calculate_dis(): self.point2_beta_l=", self.point2_beta_l)

            self.target_point1 = self.calculate_target_point(self.point1_alpha_l,
                                                             self.point1_alpha_r,
                                                             self.point1_beta_l,
                                                             self.point1_beta_r)

            print("CoordinateSystem.calculate_dis(): self.point1_alpha_r=", self.point1_alpha_r)
            print("CoordinateSystem.calculate_dis(): self.point2_alpha_r=", self.point2_alpha_r)
            print("CoordinateSystem.calculate_dis(): self.point1_beta_r=", self.point1_beta_r)
            print("CoordinateSystem.calculate_dis(): self.point2_beta_r=", self.point2_beta_r)

            self.target_point2 = self.calculate_target_point(self.point2_alpha_l,
                                                             self.point2_alpha_r,
                                                             self.point2_beta_l,
                                                             self.point2_beta_r)

            target_point1_x = self.target_point1[0]
            target_point1_y = self.target_point1[1]
            target_point1_z = self.target_point1[2]

            print("CoordinateSystem.calculate_dis(): self.target_point1= (" +
                  str(round(target_point1_x, 2)) + ", " +
                  str(round(target_point1_y, 2)) + ", " +
                  str(round(target_point1_z, 2)) + ")")

            target_point2_x = self.target_point2[0]
            target_point2_y = self.target_point2[1]
            target_point2_z = self.target_point2[2]

            print("CoordinateSystem.calculate_dis(): self.target_point2= (" +
                  str(round(target_point2_x, 2)) + ", " +
                  str(round(target_point2_y, 2)) + ", " +
                  str(round(target_point2_z, 2)) + ")")

            dis = ((target_point1_x - target_point2_x) ** 2 + (target_point1_y - target_point2_y) ** 2 + (
                        target_point1_z - target_point2_z) ** 2) ** 0.5

            return dis
        return "ERROR !!! Target point is not clicked !!!"

    def calculate_target_point(self, alpha1, alpha2, beta1, beta2):
        print("[Start] !!! calculate_target_point !!! [Start]")

        print("3", self.cam1_3d_coordinate)
        alpha1 = math.radians(alpha1)
        alpha2 = math.radians(alpha2)

        beta1_prime = 90 - beta1
        if beta1_prime >= 360:
            beta1_prime -= 360
        elif beta1_prime < 0:
            beta1_prime += 360

        beta2_prime = 90 - beta2
        if beta2_prime >= 360:
            beta2_prime -= 360
        elif beta2_prime < 0:
            beta2_prime += 360

        print("CoordinateSystem.calculate_target_point(): alpha1=", alpha1)
        print("CoordinateSystem.calculate_target_point(): alpha2=", alpha2)
        print("CoordinateSystem.calculate_target_point(): beta1=", beta1)
        print("CoordinateSystem.calculate_target_point(): beta2=", beta2)
        print("CoordinateSystem.calculate_target_point(): beta1_prime=", beta1_prime)
        print("CoordinateSystem.calculate_target_point(): beta2_prime=", beta2_prime)

        beta1_prime = math.radians(beta1_prime)
        beta2_prime = math.radians(beta2_prime)

        line1_vector_x = math.sin(alpha1) * math.cos(beta1_prime)
        line1_vector_y = math.sin(alpha1) * math.sin(beta1_prime)
        line1_vector_z = math.cos(alpha1)

        print("CoordinateSystem.calculate_target_point(): line1_vector_x=", line1_vector_x)
        print("CoordinateSystem.calculate_target_point(): line1_vector_y=", line1_vector_y)
        print("CoordinateSystem.calculate_target_point(): line1_vector_z=", line1_vector_z)

        line1_const_x = self.cam1_3d_coordinate[0]
        line1_const_y = self.cam1_3d_coordinate[1]
        line1_const_z = self.cam1_3d_coordinate[2]

        print("CoordinateSystem.calculate_target_point(): line1_const_x=", line1_const_x)
        print("CoordinateSystem.calculate_target_point(): line1_const_y=", line1_const_y)
        print("CoordinateSystem.calculate_target_point(): line1_const_z=", line1_const_z)

        s = Symbol('s')
        point_p = [line1_vector_x * s - line1_const_x,
                   line1_vector_y * s - line1_const_y,
                   line1_vector_z * s - line1_const_z]

        print("CoordinateSystem.calculate_target_point(): point_p=", point_p)

        line2_vector_x = math.sin(alpha2) * math.cos(beta2_prime)
        line2_vector_y = math.sin(alpha2) * math.sin(beta2_prime)
        line2_vector_z = math.cos(alpha2)

        print("CoordinateSystem.calculate_target_point(): line2_vector_x=", line2_vector_x)
        print("CoordinateSystem.calculate_target_point(): line2_vector_y=", line2_vector_y)
        print("CoordinateSystem.calculate_target_point(): line2_vector_z=", line2_vector_z)

        line2_const_x = self.cam2_3d_coordinate[0]
        print("line2_cos_x: {}".format(line2_const_x))
        line2_const_y = self.cam2_3d_coordinate[1]
        line2_const_z = self.cam2_3d_coordinate[2]

        print("CoordinateSystem.calculate_target_point(): line2_const_x=", line2_const_x)
        print("CoordinateSystem.calculate_target_point(): line2_const_y=", line2_const_y)
        print("CoordinateSystem.calculate_target_point(): line2_const_z=", line2_const_z)

        t = Symbol('t')
        point_q = [line2_vector_x * t - line2_const_x,
                   line2_vector_y * t - line2_const_y,
                   line2_vector_z * t - line2_const_z]

        print("CoordinateSystem.calculate_target_point(): point_q=", point_q)

        vector_pq_x = point_q[0] - point_p[0]
        vector_pq_y = point_q[1] - point_p[1]
        vector_pq_z = point_q[2] - point_p[2]

        print("CoordinateSystem.calculate_target_point(): vector_pq_x=", vector_pq_x)
        print("CoordinateSystem.calculate_target_point(): vector_pq_y=", vector_pq_y)
        print("CoordinateSystem.calculate_target_point(): vector_pq_z=", vector_pq_z)

        eq1 = Eq((vector_pq_x * line1_vector_x + vector_pq_y * line1_vector_y + vector_pq_z * line1_vector_z), 0)

        print("CoordinateSystem.calculate_target_point(): eq1=", eq1)

        eq2 = Eq((vector_pq_x * line2_vector_x + vector_pq_y * line2_vector_y + vector_pq_z * line2_vector_z), 0)

        print("CoordinateSystem.calculate_target_point(): eq2=", eq2)

        ans_s_t = solve((eq1, eq2), (s, t))

        print("CoordinateSystem.calculate_target_point(): ans_s_t=", ans_s_t)

        print("CoordinateSystem.calculate_target_point(): s=", ans_s_t[s])
        print("CoordinateSystem.calculate_target_point(): t=", ans_s_t[t])

        s = ans_s_t[s]
        t = ans_s_t[t]

        point_p_ans = [line1_vector_x * s - line1_const_x,
                       line1_vector_y * s - line1_const_y,
                       line1_vector_z * s - line1_const_z]

        point_q_ans = [line2_vector_x * t - line2_const_x,
                       line2_vector_y * t - line2_const_y,
                       line2_vector_z * t - line2_const_z]

        print("CoordinateSystem.calculate_target_point(): point_p_ans=", point_p_ans)
        print("CoordinateSystem.calculate_target_point(): point_q_ans=", point_q_ans)

        target_point = [0, 0, 0]
        target_point[0] = (point_p_ans[0] + point_q_ans[0]) / 2
        target_point[1] = (point_p_ans[1] + point_q_ans[1]) / 2
        target_point[2] = (point_p_ans[2] + point_q_ans[2]) / 2

        print("CoordinateSystem.calculate_target_point(): target_point_=", target_point)

        return target_point
