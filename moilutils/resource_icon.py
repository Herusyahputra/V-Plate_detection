"""

Project Name: MoilApps v3.1.0
Writer : Haryanto
PROJECT MADE WITH: Qt Designer and PyQt6
Build for: MOIL-LAB
Copyright: MOIL-2022

This project can be use a a template to create better user interface when you design a project.

There are limitations on Qt licenses if you want to use your products
commercially, I recommend reading them on the official website:
https://doc.qt.io/qtforpython/licenses.html

"""
import os
try:
    from PyQt6.QtCore import QDir
    pyqt_version = "pyqt6"

except:
    from PyQt5.QtCore import QDir
    pyqt_version = "pyqt5"

CURRENT_DIRECTORY = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
QDir.addSearchPath("icons", CURRENT_DIRECTORY + "/icons")
