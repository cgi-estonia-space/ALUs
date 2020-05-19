#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


/*
    By definition a cudafrienly object is capable of sending all or some of
    its internal values under pointers to the gpu.
    It does not send itself to the gpu.
*/
class CudaFriendlyObject{
public:
    virtual void hostToDevice() = 0;
    virtual void deviceToHost() = 0;
    virtual void deviceFree() = 0;
};
