import yaml


class Load_save(object):
    def __init__(self, parent):
        super(Load_save, self).__init__()
        self.parent = parent
        print("from save in")

    def save_param(self):
        """
        This function is to save parameters which have been your setting on the user interface such as set value (alpha,
        beta, zoom) used to got anypoint map x, map y image
        """
        data = [self.setup_save()]
        with open("./backend/config.yaml", 'w') as f:
            yaml.dump(data, f)
            print("save")

    def load_param(self):
        """
        This function is to load parameter with the yaml format which have been your save the previous
        """
        # file = "config.yaml"
        with open("./backend/config.yaml") as f:
            docs = yaml.load_all(f, Loader=yaml.FullLoader)
            for doc in docs:
                print(doc)
                self.setup_load(doc)

    def setup_save(self):
        """
        This function is to save the value (alpha. beta. zoom) inside and outside which on the input in the user interface
        :return:
            value data
        """
        data = {"data config": {
            "alpha_inside": self.parent.val_alpha_in.value(),
            "beta_inside": self.parent.val_beta_in.value(),
            "zoom_inside": self.parent.val_zoom_in.value(),
            "alpha_outside": self.parent.val_alpha_out.value(),
            "beta_outside": self.parent.val_beta_out.value(),
            "zoom_outside": self.parent.val_zoom_out.value()
        }}
        return data

    def setup_load(self, doc):
        """
        This function is to set load the value data
        """
        data = doc[0]["data config"]
        try:
            self.parent.blockSignals()
            self.parent.val_alpha_in.setValue(data["alpha_inside"])
            self.parent.val_beta_in.setValue(data["beta_inside"])
            self.parent.val_zoom_in.setValue(data["zoom_inside"])
            self.parent.val_alpha_out.setValue(data["alpha_outside"])
            self.parent.val_beta_out.setValue(data["beta_outside"])
            self.parent.val_zoom_out.setValue(data["zoom_outside"])
            self.parent.unblockSignals()

        except:
            print("no parameter")
