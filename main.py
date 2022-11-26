import sys
from PyQt6 import QtWidgets
# from controller import controller
from controller.controller_only_plate import controller


def main():
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = controller(MainWindow)
    # ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()