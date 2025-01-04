#include <iostream>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;


//we will be using struct datatype to contain all the properties of the sphere and also the intersection checker so that we can call it under Sphere 

struct Structure {
  glm::vec3 color ; //first the property => color
  glm::vec3 center ; //second the property => center
  float radius ; //third the property => radius
  __device__ bool intersect_checker(const glm::vec3 ray_origin,glm::vec3 ray_dir){//It takes input =>ray_origin
    glm::vec3 OC = ray_origin - center;
    float b = glm::dot(OC,ray_dir); //It calculates just coefficient{b} of the quad
    float c = glm::dot(OC,OC)-radius*radius; //It calculates just coefficient{c} of the quad
    float discriminant = b*b - c;
    float t =0.0f;
    if (discriminant > 0){
     t=-b-sqrt(discriminant);
     if(t>0.001f) return true;
    }
    return false;
  }
};

//we will have a rendering kernel to fill the positions on the screen than faces intersectionwith the correct colour data in a flat array later to be used for projecting the image 

__global__ void rendering_kernel (Structure *structure,int Dim,const glm::vec3 ray_origin,glm::vec3 *d_image){
  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  glm::vec3 ray_dir = glm::normalize(glm::vec3((x - (Dim) / 2) / float(Dim), (y - (Dim) / 2) / float(Dim), -1.0f));

  if(x<Dim&&y<Dim){
     bool status = structure->intersect_checker(ray_origin,ray_dir);
     if(status){
       int pixel_index = y*Dim+x;
       d_image[pixel_index] = structure->color;
     }   
  }
}   

int main(){
   //Hard coding the dimension of the screen and the ray_origin
    int Dim=600;
    glm::vec3 *ray_origin = new glm::vec3(0.0f,0.0f,0.0f);
   //we harcode the sphere properties center,radius and color  
    Structure sphere_0 ;
    sphere_0.color = glm::vec3(1.0f,0.0f,0.0f);
    sphere_0.center = glm::vec3(0.0f,0.0f,-5.0f);
    sphere_0.radius = 1.0f;
     
    Structure sphere_1 ;
    sphere_1.color = glm::vec3(0.0f,1.0f,0.0f);
    sphere_1.center = glm::vec3(1.0f,-1.0f,-5.0f);
    sphere_1.radius = 1.0f;
    
    Structure sphere_2 ;
    sphere_2.color = glm::vec3(0.0f,0.0f,1.0f);
    sphere_2.center = glm::vec3(-1.0f,1.0f,-5.0f);
    sphere_2.radius = 1.0f;
   //allocate memory for the d_image 
    glm::vec3 *d_image ;
    cudaMalloc((void**)&d_image,Dim*Dim*sizeof(glm::vec3));
   //copy the spheres to the device
    Structure *d_spheres[3]; 
    cudaMalloc((void**)&d_spheres[0],sizeof(Structure));
    cudaMalloc((void**)&d_spheres[1],sizeof(Structure));
    cudaMalloc((void**)&d_spheres[2],sizeof(Structure));
    cudaMemcpy(d_spheres[0],&sphere_0,sizeof(Structure),cudaMemcpyHostToDevice);
    cudaMemcpy(d_spheres[1],&sphere_1,sizeof(Structure),cudaMemcpyHostToDevice);
    cudaMemcpy(d_spheres[2],&sphere_2,sizeof(Structure),cudaMemcpyHostToDevice);
   //Launch the kernel
    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid(Dim/threadsPerBlock.x,Dim/threadsPerBlock.y);
    for (int i=0;i<3;++i){
     rendering_kernel <<< blocksPerGrid,threadsPerBlock >>> (d_spheres[i],Dim,*ray_origin,d_image);
     cudaDeviceSynchronize();
    }
   //intiate vectore store in the host
    glm::vec3 *h_image = new glm::vec3[Dim*Dim];
    cudaMemcpy(h_image,d_image,Dim*Dim*sizeof(glm::vec3),cudaMemcpyDeviceToHost);
   //we need to save the data in an image format
    unsigned char *pixels = new unsigned char[Dim*Dim*3];
    for(int i=0;i<Dim*Dim;++i){
     pixels[3*i] = static_cast<unsigned char>(glm::clamp(h_image[i].r*255.0f,0.0f,255.0f));
     pixels[3*i+1]= static_cast<unsigned char>(glm::clamp(h_image[i].g*255.0f,0.0f,255.0f));
     pixels[3*i+2]= static_cast<unsigned char>(glm::clamp(h_image[i].b*255.0f,0.0f,255.0f));
    }
   //we need to save the image now [using png format]
    stbi_write_png("output_1.png",Dim,Dim,3,pixels,Dim*3);
   //Clean up
    delete[] pixels;
    delete[] h_image;
    delete[] ray_origin;
 
    cudaFree(d_spheres[0]);cudaFree(d_spheres[1]);cudaFree(d_spheres[2]);
    cudaFree(d_image);
    return 0;
}
