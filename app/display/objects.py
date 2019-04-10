from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np


class OBJ:
    def __init__(self, fdir, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []

        self.mtl=None

        material = None
        for line in open(fdir+filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                #v = map(float, values[1:4])
                v=[ float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                #v = map(float, values[1:4])
                v=[ float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                v = [float(x) for x in values[1:3]]

                self.texcoords.append(v)
            elif values[0] in ('usemtl', 'usemat') and len(values) > 1:
                material = values[1]
            elif values[0] == 'mtllib':
                #print(values[1])
                #self.mtl = MTL(fdir,values[1])
                self.mtl = [fdir,values[1]]
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))

    def create_gl_list(self):
        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)
        # glCullFace(GL_BACK)
        # glEnable(GL_CULL_FACE)

        for face in self.faces:
            vertices, normals, texture_coords, material = face

            # mtl = self.mtl[material]
            # if 'texture_Kd' in mtl:
            #     # use diffuse texmap
            #     glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])
            # else:
            #     # just use diffuse colour
            #     # print(mtl['Kd'],"----")
            #     glColor(*mtl['Kd'])

            glBegin(GL_POLYGON)
            for i in range(len(vertices)):
                if normals[i] > 0:
                    glNormal3fv(self.normals[normals[i] - 1])
                if texture_coords[i] > 0:
                    glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
                glVertex3fv(self.vertices[vertices[i] - 1])
            glEnd()
        glDisable(GL_TEXTURE_2D)
        glEndList()


# obj = OBJ('../smpl/', 'smpl_np_3.obj')
# obj = OBJ('./test/beta-objs/', '0.obj')

if __name__ == '__main__':
    pass
