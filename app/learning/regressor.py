from app.geometry.mesh import Mesh
import numpy as np
from app.learning.mlp import disp
from app.geometry.smooth import smooth


class Regressor:
    displacement = np.array([])

    def apply(self, mesh):
        mesh.vertices += self.displacement
        return mesh


if __name__ == '__main__':
    rg = Regressor()
    rg.displacement = np.reshape(disp, (17436, 3))
    smooth(rg.apply(Mesh().load('../data/beta_simulation/avg_smooth.obj')), 0).save('../test/rebuild1.obj')