#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class CudaFriendlyObject{
public:
    virtual void hostToDevice() = 0;
    virtual void deviceToHost() = 0;
    virtual void deviceFree() = 0;
};
