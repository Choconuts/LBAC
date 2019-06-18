from lbac.display.regressor import PoseRegressor, ShapeRegressor
from com.learning.canvas import Canvas
from com.path_helper import *


def test_gru():

    gru_canvas = Canvas()

    pr = PoseRegressor(gru_canvas, conf_path('model/sgru/1'), 5)

    res = pr.gen(np.zeros(10), np.zeros((24,3)))

    print(np.mean(np.square(res)))

    gru_canvas.close()


def test_mlp():
    mlp_canvas = Canvas()

    sr = ShapeRegressor(mlp_canvas, conf_path('model/mlp/1'))

    res = sr.gen(np.zeros(10))

    print(np.mean(np.square(res)))

    mlp_canvas.close()


if __name__ == '__main__':
    """
    """
    test_mlp()






