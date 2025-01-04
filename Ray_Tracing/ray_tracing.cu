#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"  

// Basic structure for a sphere
struct Sphere {
    glm::vec3 center;
    float radius;
    glm::vec3 color;

    __device__ bool intersect(const glm::vec3 &ray_origin, const glm::vec3 &ray_dir, float &t) {
        glm::vec3 oc = ray_origin - center;
        float b = glm::dot(oc, ray_dir);
        float c = glm::dot(oc, oc) - radius * radius;
        float discriminant = b * b - c;
        if (discriminant > 0) {
            t = -b - sqrtf(discriminant);
            if (t > 0.001f) return true;
        }
        return false;
    }
};

// Render function (kernel) to trace rays
__global__ void renderKernel(Sphere *spheres, int sphere_count, glm::vec3 *d_image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    glm::vec3 ray_origin(0.0f, 0.0f, 0.0f);  // Camera origin
    glm::vec3 ray_dir = glm::normalize(glm::vec3((x - width / 2) / float(width), (y - height / 2) / float(height), -1.0f));  // Ray direction

    glm::vec3 color(0.0f);  // Background color

    // Check intersections with spheres
    for (int i = 0; i < sphere_count; ++i) {
        float t = 0;
        if (spheres[i].intersect(ray_origin, ray_dir, t)) {
            color = spheres[i].color;  // Color of the first intersected sphere
        }
    }

    int pixel_index = y * width + x;
    d_image[pixel_index] = color;
}

void renderImage(int width, int height) {
    // Create a simple scene with spheres
    std::vector<Sphere> spheres = {
        { glm::vec3(0.0f, 0.0f, -5.0f), 1.0f, glm::vec3(1.0f, 0.0f, 0.0f) },  // Red sphere
        { glm::vec3(2.0f, 0.0f, -6.0f), 1.0f, glm::vec3(0.0f, 1.0f, 0.0f) },  // Green sphere
        { glm::vec3(-2.0f, 0.0f, -6.0f), 1.0f, glm::vec3(0.0f, 0.0f, 1.0f) }  // Blue sphere
    };

    // Allocate memory for spheres on the device
    Sphere *d_spheres;
    cudaMalloc((void**)&d_spheres, spheres.size() * sizeof(Sphere));
    cudaMemcpy(d_spheres, spheres.data(), spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice);

    // Allocate memory for image on the device
    glm::vec3 *d_image;
    cudaMalloc((void**)&d_image, width * height * sizeof(glm::vec3));

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    renderKernel<<<gridSize, blockSize>>>(d_spheres, spheres.size(), d_image, width, height);

    // Copy result back to host
    glm::vec3 *h_image = new glm::vec3[width * height];
    cudaMemcpy(h_image, d_image, width * height * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    // Convert to 8-bit RGB values
    unsigned char *pixels = new unsigned char[width * height * 3];  // 3 bytes per pixel (RGB)
    for (int i = 0; i < width * height; ++i) {
        pixels[3 * i] = static_cast<unsigned char>(glm::clamp(h_image[i].r * 255.0f, 0.0f, 255.0f));
        pixels[3 * i + 1] = static_cast<unsigned char>(glm::clamp(h_image[i].g * 255.0f, 0.0f, 255.0f));
        pixels[3 * i + 2] = static_cast<unsigned char>(glm::clamp(h_image[i].b * 255.0f, 0.0f, 255.0f));
    }

    // Save the image as a PNG file using stb_image_write
    stbi_write_png("output.png", width, height, 3, pixels, width * 3);

    // Cleanup
    delete[] pixels;
    delete[] h_image;
    cudaFree(d_spheres);
    cudaFree(d_image);
}

int main() {
    int width = 800;  // Image width
    int height = 600; // Image height

    renderImage(width, height);

    std::cout << "Rendering completed! Image saved as 'output.png'." << std::endl;
    return 0;
}
