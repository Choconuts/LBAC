from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import  numpy as np
from app.geometry.mesh import Mesh
# from PyOpenGLtoolbox import *

class Shader:

    def __init__(self):
        self.id = -1
        self.location = dict()
        pass

    def init(self):
        pass

    def use(self):
        pass


class SimpleShader(Shader):
    vertex_shader_string = '''
                varying vec3 normal;
                attribute vec3 a_normal;   //位置是0
                attribute vec3 position;  //位置是1
                varying float id;
                attribute float a_id;
                void main() {
                    normal = a_normal;
                    vec4 h_normal = gl_ModelViewProjectionMatrix * vec4(a_normal, 1);
                    normal = h_normal.xyz;
                    gl_Position = gl_ModelViewProjectionMatrix * vec4(position, 1);
                    id = a_id;
                }
            '''

    fragment_shader_string = '''
                varying vec3 normal;
                uniform vec3 color_0;
                uniform vec3 color_1;
                uniform vec3 color_2;
                varying float id;
                void main() {
                    float intensity;
                    vec4 color;
                    vec3 n = normalize(normal);
                    //vec3 l = normalize(gl_LightSource[0].position).xyz;
                    vec3 l = vec3(0.2, 1, -0.5);
                    
                    intensity = dot(l, n);
                    if (intensity < 0.0) intensity = 0.0;
                    if (id < 0.9)
                        color = vec4(color_0 * intensity,1);
                    else if (id < 1.9)
                        color = vec4(color_1 * intensity,1);
                    else if (id < 2.9)
                        color = vec4(color_2 * intensity,1);
                    color = color * 0.9 + vec4(1, 1, 1, 1) * 0.1;

                    gl_FragColor = color;
                }
            '''

    def __init__(self):
        Shader.__init__(self)
        self.init()
        self.color(0, [1, 1, 1])
        self.color(1, [1, 1, 1])

    def init(self):
        self.id = compile_shader(self.vertex_shader_string, self.fragment_shader_string)
        self.use()
        self.location['position'] = glGetAttribLocation(self.id, 'position')
        self.location['normal'] = glGetAttribLocation(self.id, 'a_normal')
        self.location['id'] = glGetAttribLocation(self.id, 'a_id')
        self.location['color'] = []
        for i in range(3):
            self.location['color'].append(glGetUniformLocation(self.id, 'color_' + str(i)))
        glVertexAttribPointer(self.location['position'], 3, GL_FLOAT, GL_FALSE, 7 * 4, c_void_p(0))  # 这里的None不能写为0
        glVertexAttribPointer(self.location['normal'], 3, GL_FLOAT, GL_TRUE, 7 * 4, c_void_p(3 * 4))  # 这里的None不能写为0
        glVertexAttribPointer(self.location['id'], 1, GL_FLOAT, GL_FALSE, 7 * 4, c_void_p(6 * 4))  # 这里的None不能写为0
        return self

    def use(self):
        glUseProgram(self.id)
        return self

    def color(self, i, rgb):
        glUniform3fv(self.location['color'][i], 1, np.array(rgb, np.float32))
        return self

    def draw(self):
        glEnableVertexAttribArray(self.location['position'])
        glEnableVertexAttribArray(self.location['normal'])
        glEnableVertexAttribArray(self.location['id'])


def create_shader(ShaderProgram, Shader_Type, Source):  #创建并且添加着色器（相当于AddShader）Shader_Type为类型
    ShaderObj = glCreateShader( Shader_Type )  #创建Shader对象
    glShaderSource(ShaderObj , Source)
    glCompileShader(ShaderObj)  #进行编译
    glAttachShader(ShaderProgram, ShaderObj)  #将着色器对象关联到程序上


def compile_shader(VERTEX_SHADER, FRAGMENT_SHADER):  #编译着色器
    Shader_Program = glCreateProgram()  #创建空的着色器程序
    create_shader(Shader_Program, GL_VERTEX_SHADER, VERTEX_SHADER)
    create_shader(Shader_Program, GL_FRAGMENT_SHADER, FRAGMENT_SHADER)
    glLinkProgram(Shader_Program)
    glUseProgram(Shader_Program)
    return Shader_Program

