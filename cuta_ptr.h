#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <iostream>

constexpr int length = 20;

template <typename Deleter>
using unique_p = std::unique_ptr<float[], Deleter>;

struct myDeleter{
    void operator()(float* ptr){ cudaFree(ptr); std::cout<<"\nDeleted\n"; } 
};

void version3(){
    auto  myCudaMalloc = [](size_t mySize) { void* ptr; cudaMalloc((void**)&ptr, mySize); return ptr; };
    auto deleter = [](float* ptr) { cudaFree(ptr); std::cout<<"\nDeleted3\n"; };
    unique_p<decltype(deleter)> d_in((float*)myCudaMalloc(length*sizeof(float)),deleter);
    //unique_p<myDeleter> d_in((float*)myCudaMalloc(20*sizeof(float)));

    std::unique_ptr<float[]> h_out(new float[length]);
    for(int i = 0; i < length; i++){ h_out[i] = i; }

    cudaMemcpy(d_in.get(), h_out.get(),length*sizeof(float),cudaMemcpyHostToDevice);

    printArray<<<1,1>>>(d_in.get(),length);
}