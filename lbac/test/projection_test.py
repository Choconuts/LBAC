import pytest
from com.mesh.projection import *
from com.mesh.lab_3d import *
from com.mesh.mesh import *
from com.posture.smpl import *
from lbac.test.sequence_test import *
from lbac.test.pose_gt_test import *


p = vec3(0.5, 0.3, 1)
tri = [vec3(1, 0.5, 0), vec3(0, 1, 0.5), vec3(1, 1, 1)]


def test_projection():
    plane = Plane().from_triangle(*tri)
    vec = point_plane_project_vec(p, plane)

    contact = p + vec
    uu, vv = solve_uv(p, *tri)

    print(uu, vv)

    re = (1.01 - uu - vv) * tri[0] + uu * tri[1] + vv * tri[2]
    print(re)

    lab = Lab()
    lab.color([0.5, 0.8, 0.6])
    lab.add_triangle(tri)
    lab.color([0.7, 0.4, 0.8])
    lab.add_line(p, contact)
    lab.color([0.9, 0.6, 0.4])
    lab.add_point(p)
    lab.add_point(contact)
    lab.add_point(re)
    # lab.graphic()
    print(vec)


if __name__ == '__main__':
    pass