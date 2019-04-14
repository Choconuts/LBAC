from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import  numpy as np
from app.geometry.mesh import Mesh
# from PyOpenGLtoolbox import *
from app.display.shader.shaders import SimpleShader
from app.display.vbo import StaticVBO
from app.display.simple_display import VertexArray

obj = Mesh().load('../test/save_mesh的副本 4.obj')

gpgpu_v = """
        //#version 330 compatibility
        #version 130
        out vec4 pos;
        void main() {  
            pos = vec4(gl_Vertex);
            //The following coding must be in fragment shader
            //vec2 xy = v.xy;
            //vec2 uv = vec2(xy/vec2(xw,yw)).xy;
            //o_color = texture2D(tex0, uv);
            gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
        }
        """
gpu_v = """
        #version 330 core
        Layout (location = 0) in ver3 aPos;
        void main()
        {
          gl_Position = ver4(aPos.x, aPos.y, aPos.z, 1.0);
        }
        """

gpgpu_f = """
        //#version 330 compatibility
        #version 130
        in vec4 pos;
        uniform sampler2D tex0; 
        uniform float xw;
        uniform float yw;          
        void main() {             
            vec2 xy = pos.xy;
            vec2 uv = vec2(xy/vec2(xw,yw)).xy;            
            vec4 o_color = texture2D(tex0, uv);// vec4(uv.x,uv.y, 0, 1 );//   
            o_color = o_color + vec4(0.2);  
            gl_FragColor = o_color;
        }"""

gpu_f = """
        #version 330 core
        out vec4 FragColor;
        void main()
        {
          // RGBA         r   g    b   a
        FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f); 
        }
        """


VERTEX_SHADER = '''
            varying vec3 normal;
            attribute vec3 a_normal;   //位置是0
            attribute vec3 position;  //位置是1
            void main() {
                normal = a_normal;
                vec4 h_normal = gl_ModelViewProjectionMatrix * vec4(a_normal, 1);
                normal = h_normal.xyz;
                gl_Position = gl_ModelViewProjectionMatrix * vec4(position, 1);
            }
        '''
#片段着色器部分,字符串类型
FRAGMENT_SHADER = '''
            varying vec3 normal;
            uniform vec3 a_color;
            void main() {
                float intensity;
                vec4 color;
                vec3 n = normalize(normal);
                //vec3 l = normalize(gl_LightSource[0].position).xyz;
                vec3 l = vec3(0.2, 1, -0.5);

                // quantize to 5 steps (0, .25, .5, .75 and 1)
                intensity = dot(l, n);
                if (intensity < 0.0) intensity = 0.0;
                color = vec4(a_color * intensity,1);
                color = color * 0.9 + vec4(1, 1, 1, 1) * 0.1;

                gl_FragColor = color;
            }
    '''
#
# '''
#             varying vec3 normal;
#             void main() {
#                 normal = gl_NormalMatrix * gl_Normal;
#                 gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
#             }
#         '''

# '''
#             varying vec3 normal;
#             void main() {
#                 float intensity;
#                 vec4 color;
#                 vec3 n = normalize(normal);
#                 vec3 l = normalize(gl_LightSource[0].position).xyz;
#
#                 // quantize to 5 steps (0, .25, .5, .75 and 1)
#                 intensity = (floor(dot(l, n) * 4.0) + 1.0)/4.0;
#                 color = vec4(intensity*1.0, intensity*0.5, intensity*0.5,
#                     intensity*1.0);
#
#                 gl_FragColor = color;
#             }
#     '''


vertices = [
  -0.5,   0.0,  0.0,
   0.5,   0.0,  0.0,
   0.0,  0.86,  0.0,
]

vertices = np.array(vertices)

vao = 0
shaderPrograme = 0

def Create_Shader( ShaderProgram, Shader_Type , Source):  #创建并且添加着色器（相当于AddShader）Shader_Type为类型
    ShaderObj = glCreateShader( Shader_Type )  #创建Shader对象
    glShaderSource(ShaderObj , Source)
    glCompileShader(ShaderObj)  #进行编译
    glAttachShader(ShaderProgram, ShaderObj)  #将着色器对象关联到程序上


def Compile_Shader():  #编译着色器
    Shader_Program = glCreateProgram()  #创建空的着色器程序
    Create_Shader(Shader_Program , GL_VERTEX_SHADER , VERTEX_SHADER)
    Create_Shader(Shader_Program , GL_FRAGMENT_SHADER , FRAGMENT_SHADER)
    glLinkProgram(Shader_Program)
    glUseProgram(Shader_Program)
    color_location = glGetAttribLocation(
        Shader_Program, 'a_normal'
    )
    print(color_location)
    color_location = glGetAttribLocation(
        Shader_Program, 'position'
    )
    print(color_location)
    print(type(Shader_Program))

rot = 0
def Draw():
    global rot
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    # glEnableVertexAttribArray(0)
    # glEnableVertexAttribArray(1)
    SimpleShader.draw()
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    # 变量位置，变量数量，变量类型，是否normalize，变量间隔，变量偏移
    # glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, c_void_p(0)) # 这里的None不能写为0
    # glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, c_void_p(3 * 4))  # 这里的None不能写为0

    glPushMatrix()
    glScale(0.5, 0.5, 0.5)
    glRotatef(rot, 0, 1, 0)
    rot += 1
    glDrawArrays(GL_TRIANGLES, 0, 6)
    glPopMatrix()
    glDisableVertexAttribArray(0)  # 解析数据 例如一个矩阵里含有 位置 、颜色、多种信息
    glutSwapBuffers()


def CreateBuffer():  #创建顶点缓存器
    global vbo   #设置为全局变量
    vertex = np.array([-0.5, 0.0, 1.0, -0.5, 0, -0.5,
                       0.0, -1.0, 0.0, -0.5, 0, -0.5,
                       0.0, 1.0, 0.0, -0.5, 0, -0.5,
                       0.0, 1.0, 0.0, 0.5, 0, -0.5,
                       0.0, -1.0, 0.0, 0.5, 0, -0.5,
                       0.5, 0.0, 1.0, 0.5, 0, -0.5
                       ], dtype="float32")  # 创建顶点数组
    vbo = glGenBuffers(1)  #创建缓存
    glBindBuffer(GL_ARRAY_BUFFER, vbo)   #绑定
    glBufferData(GL_ARRAY_BUFFER , vertex.nbytes, vertex, GL_STATIC_DRAW)   #输入数据


def tes_draw():
    global rot
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    shader.draw()

    glPushMatrix()
    glScale(1.5, 1.5, 1.5)
    glRotatef(rot, 0, 1, 0)
    rot += 1
    glBindBuffer(GL_ARRAY_BUFFER, vbo2.id)
    glDrawArrays(GL_TRIANGLES, 0, int(vbo2.num / 7))
    # glBindBuffer(GL_ARRAY_BUFFER, vbo3.id)
    # glDrawArrays(GL_TRIANGLES, 0, int(vbo3.num / 6))
    glPopMatrix()
    glDisableVertexAttribArray(0)  # 解析数据 例如一个矩阵里含有 位置 、颜色、多种信息
    glutSwapBuffers()


def main():
    glutInit([])
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)  # 显示模式 双缓存
    glutInitWindowPosition(100, 100)  # 窗口位置
    glutInitWindowSize(500, 500)  # 窗口大小
    glutInitContextVersion(4,3)   #为了兼容
    glutInitContextProfile(GLUT_CORE_PROFILE)   #为了兼容
    # glutDisplayFunc(Draw)  # 回调函数
    # glutIdleFunc(Draw)  # 回调函数
    # glClearColor(0.0, 0.0, 0.0, 0.0)
    # CreateBuffer()
    # SimpleShader(VERTEX_SHADER, FRAGMENT_SHADER).init()
    # glEnable(GL_DEPTH_TEST)
    # glutMainLoop()
    global vbo2, vbo3
    # vbo2 = StaticVBO().bind(np.array([-0.5, 0.0, 1.0, -0.5, 0, -0.5,
    #                    0.0, -1.0, 0.0, -0.5, 0, -0.5,
    #                    0.0, 1.0, 0.0, -0.5, 0, -0.5,
    #                    0.0, 1.0, 0.0, 0.5, 0, -0.5,
    #                    0.0, -1.0, 0.0, 0.5, 0, -0.5,
    #                    0.5, 0.0, 1.0, 0.5, 0, -0.5
    #                    ], dtype="float32"))
    verts = Mesh().load('./../test/anima/seq1/1.obj').to_vertex_buffer() # './../smpl/17-bodies/1.obj'
    verts = VertexArray(verts).add_cols([1]).get()
    verts2 = Mesh().load('./../smpl/17-bodies/1.obj').to_vertex_buffer()
    verts2 = VertexArray(verts2).add_cols([0]).get()

    glutCreateWindow("sanjiao")  # 创建窗口
    glutDisplayFunc(tes_draw)  # 回调函数
    glutIdleFunc(tes_draw)  # 回调函数

    vbo2 = StaticVBO().bind(np.hstack((verts, verts2)))
    global shader
    shader = SimpleShader().color(0, [1, 0, 0])
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glutMainLoop()


def init_shader():
    pass

if __name__ == '__main__':
     main()
