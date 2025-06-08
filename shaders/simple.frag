#version 430 core

in vec4 vertexColor;  // Interpolated color from vertex shader
in vec3 vertexNormal; // Interpolated normal from vertex shader

out vec4 color;

void main()
{
    // Define the light direction (normalized)
    vec3 lightDirection = normalize(vec3(0.8, -0.5, 0.6)); // Sunlight direction

    // Normalize the vertex normal to ensure proper lighting calculation
    vec3 normalizedNormal = normalize(vertexNormal);

    // Lambertian diffuse lighting model
    float lightIntensity = max(dot(normalizedNormal, -lightDirection), 0.0);

    // Apply the lighting to the vertex color (only affect RGB, not alpha)
    vec3 litColor = vertexColor.rgb * lightIntensity;

    color = vec4(litColor, vertexColor.a); // Set the final fragment color with lighting
}
