#version 430 core

layout(location = 0) in vec3 position; // Vertex position
layout(location = 1) in vec4 color;    // Vertex color
layout(location = 2) in vec3 normal;   // Vertex normal

out vec4 vertexColor;  // Pass color to fragment shader
out vec3 vertexNormal; // Pass transformed normal to fragment shader

uniform mat4 transform; // 4x4 transformation matrix
uniform mat3 normalMatrix; // 3x3 matrix to correctly transform normals

void main()
{
    // Transform position using the full 4x4 transformation matrix
    gl_Position = transform * vec4(position, 1.0);

    // Pass the vertex color through
    vertexColor = color;

    // Transform the normal using the normal matrix
    vertexNormal = normalize(normalMatrix * normal);
}
