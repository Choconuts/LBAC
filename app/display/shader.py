from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import  numpy as np
from app.geometry.mesh import Mesh

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
            void main() {
                normal = gl_NormalMatrix * gl_Normal;
                gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
            }
        '''
#片段着色器部分,字符串类型
FRAGMENT_SHADER = '''
            varying vec3 normal;
            void main() {
                float intensity;
                vec4 color;
                vec3 n = normalize(normal);
                vec3 l = normalize(gl_LightSource[0].position).xyz;

                // quantize to 5 steps (0, .25, .5, .75 and 1)
                intensity = (floor(dot(l, n) * 4.0) + 1.0)/4.0;
                color = vec4(intensity*1.0, intensity*0.5, intensity*0.5,
                    intensity*1.0);

                gl_FragColor = color;
            }
    '''


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


def Draw():
    glClear(GL_COLOR_BUFFER_BIT)
    glEnableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None) #这里的None不能写为0
    glDrawArrays(GL_TRIANGLES, 0, 3)
    glDisableVertexAttribArray(0)  #解析数据 例如一个矩阵里含有 位置 、颜色、多种信息
    glutSwapBuffers()


def CreateBuffer():  #创建顶点缓存器
    global VBO   #设置为全局变量
    vertex = np.array([[-1.0,-1.0,0.0],
                       [1.0,-1.0,0.0],
                       [0.0,1.0,0.0]],dtype="float32")   #创建顶点数组
    VBO = glGenBuffers(1)  #创建缓存
    glBindBuffer(GL_ARRAY_BUFFER , VBO)   #绑定
    glBufferData(GL_ARRAY_BUFFER , vertex.nbytes , vertex , GL_STATIC_DRAW)   #输入数据


def shader_init():
    global vao, shaderPrograme
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW)

    # gen shader
    v_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(v_shader, gpu_v)
    glCompileShader(v_shader)
    f_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(f_shader, gpu_f)
    glCompileShader(f_shader)

    # link shader
    shaderPrograme = glCreateProgram()
    glAttachShader(shaderPrograme, v_shader)
    glAttachShader(shaderPrograme, f_shader)
    glLinkProgram(shaderPrograme)

    # delete shader
    glDeleteShader(v_shader)
    glDeleteShader(f_shader)
    a = 0
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3, c_void_p(a))
    glEnableVertexAttribArray(0)

    #vao
    # vao = glBindVertexArray(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    # glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW)
    # glUseProgram(shaderPrograme)
    # glBindVertexArray(vao)

def shader_test():
    # glUseProgram(shaderPrograme)
    # glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, 3)


def main():
    glutInit([])
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)  # 显示模式 双缓存
    glutInitWindowPosition(100, 100)  # 窗口位置
    glutInitWindowSize(500, 500)  # 窗口大小
    glutCreateWindow("sanjiao")  # 创建窗口
    glutInitContextVersion(4,3)   #为了兼容
    glutInitContextProfile(GLUT_CORE_PROFILE)   #为了兼容
    glutDisplayFunc(Draw)  # 回调函数
    glClearColor(0.0, 0.0, 0.0, 0.0)
    CreateBuffer()
    Compile_Shader()
    glutMainLoop()


def init_shader():
    pass

if __name__ == '__main__':
     main()