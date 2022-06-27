from tensorflow.keras.utils import plot_model

class plot_my_model:
    """Plot models using the standard plot_model from keras.util, with preset arguments
    
    Preset arguments: dpi and show_shapes
    """

    def __init__(self, show_shapes=True, expand_nested=False, dpi=56):
        self.show_shapes = show_shapes
        self.expand_nested = expand_nested
        self.dpi = dpi

    def __call__(self, model):
        return plot_model(model, show_shapes=self.show_shapes, expand_nested=self.expand_nested, dpi=self.dpi)

