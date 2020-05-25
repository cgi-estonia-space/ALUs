#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace alus{
namespace cuda{
/*
    A cuda friendly object sends some or all of it's internal pointers to the gpu and in some cases
    creates a special struct with its values and makes that available on the gpu too.
*/
class CudaFriendlyObject{
public:
    virtual void hostToDevice() = 0;
    virtual void deviceToHost() = 0;
    virtual void deviceFree() = 0;
};

}//namespace
}//namespace
