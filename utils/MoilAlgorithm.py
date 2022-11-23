import json
import math
from bisect import bisect

import numpy as np


class MoilAlgorithm:
    """
    def get_alpha_skc(self, x, y):
        clicked_x = x
        clicked_y = y

        print("MoilAlgorithm.get_alpha_skc(): clicked_x=", clicked_x)
        print("MoilAlgorithm.get_alpha_skc(): clicked_y=", clicked_y)

        delta_x = self.label_center_x - x
        delta_y = self.label_center_y - y

        iv_origin = (delta_x ** 2 + delta_y ** 2) ** 0.5
        print("MoilAlgorithm.get_alpha_skc(): iv_origin=", iv_origin)

        iv_label2calibration = iv_origin * self.ratio_label2calibration
        print("MoilAlgorithm.get_alpha_skc(): iv_label2calibration=", iv_label2calibration)

        iv, alpha_idx = self.closest(self.iv_table, iv_label2calibration)

        return math.degrees(self.alpha_table[alpha_idx])
    """

    @classmethod
    def get_alpha_griffey(cls, delta_x, delta_y, ratio_label2org):
        """

        Args:
            delta_x ():
            delta_y ():
            ratio_label2org ():

        Returns:

        """

        # print("MoilAlgorithm.get_alpha_griffey(): ratio_label2org=", ratio_label2org)
        iv_org = (delta_x ** 2 + delta_y ** 2) ** 0.5 * ratio_label2org

        # # Intel T265
        alpha = 0.00000000187574315519 * iv_org ** 4 - \
                0.00000102951859660104 * iv_org ** 3 + \
                0.00016410364525354400 * iv_org ** 2 + \
                0.18436718183957600000 * iv_org

        # Enthaniya
        # alpha = 0.00000000026290322208 * iv_org ** 5 - \
        #         0.00000006348163970275 * iv_org ** 4 + \
        #         0.00000204136367280763 * iv_org ** 3 + \
        #         0.00698944821073201000 * iv_org ** 2 + \
        #         0.00145975992329370000 * iv_org

        # alpha = 0.193769324421608 * iv_org

        # print("MoilAlgorithm.get_alpha_griffey(): ratio_cali2org=", ratio_cali2org)
        # Rasp pi
        """
        alpha = 0.000000000000000297136570437078 * iv_org ** 5 - \
                0.000000000000243083306863011000 * iv_org ** 4 + \
                0.000000000045661253803615900000 * iv_org ** 3 + \
                0.000000143872444777204000000000 * iv_org ** 2 + \
                0.000711558875950757000000000000 * iv_org / math.pi * 180"""

        print("MoilAlgorithm.get_alpha_griffey(): iv_org=", iv_org)
        print("MoilAlgorithm.get_alpha_griffey(): alpha=", alpha)

        return alpha

    @classmethod
    def get_beta(cls, delta_x, delta_y):
        """

        Args:
            delta_x ():
            delta_y ():

        Returns:

        """
        beta = 90 - math.degrees(math.atan2(delta_y, delta_x))
        if beta < 0:
            beta += 360
        print("MoilAlgorithm.get_beta(): beta=", beta)

        return beta
