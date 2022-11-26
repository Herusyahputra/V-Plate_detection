# class TestFather:
#     # def __init__(self):
#     fromFather = "I am father"
#
#     def __init__(self):
#         self.fromFather_2 = " I am from father 2"
#
#     def test_father(self, b=None):
#         if b is not None:
#             self.fromFather_2 = b
#         return self.fromFather_2
#
#
# class Test(TestFather):
#     def __init__(self, a):
#         self.a = a
#         # self.run()
#         self.a_ = TestFather()
#
#     def test_print(self):
#         self.a = self.fromFather
#         b = self.a_.test_father()
#         print(b)
#         print(self.a)
#
#     def printaPlusB(self, b):
#         self.a = b
#         print(self.a)
#         c = self.a + b
#         print(c)
#
#
# a = Test(5)
# a.test_print()
# a.printaPlusB(8)
# # print(f"test print %.1f" %(15.000))

# import numpy as np
# print(np.identity(4))

from shapely.geometry import Polygon

a = [1084, 814.0]
print((a))
print(type(([a, [1505, 814], [1505, 1130], [1084, 1130]])))
P = Polygon([a, [1505, 814], [1505, 1130], [1084, 1130]])
P = P.centroid

print((P))

from shapely.geometry import Polygon

centroid = list(Polygon([a, [1505, 814], [1505, 1130], [1084, 1130]]).centroid.coords)
print((centroid[0][0]))
# [(0.5, 0.5)]

# POINT (0.5 0.5)

# list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# print(list[-1])
