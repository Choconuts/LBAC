from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import  numpy as np

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


vertices = [
  -0.5,   0.0,  0.0,
   0.5,   0.0,  0.0,
   0.0,  0.86,  0.0,
]

vertices = np.array(vertices)

vao = 0
shaderPrograme = 0


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


def init_shader():
    pass

if __name__ == '__main__':
    pass