// This file is generated.  Any changes you make will be lost during the next clean build.

// CUDA public interface, for type definitions and api function prototypes
#include "cuda_runtime_api.h"

// *************************************************************************
//      Definitions of structs to hold parameters for each function
// *************************************************************************

// Currently used parameter trace structures 
typedef struct cudaDeviceSetLimit_v3020_params_st {
    enum cudaLimit limit;
    size_t value;
} cudaDeviceSetLimit_v3020_params;

typedef struct cudaDeviceGetLimit_v3020_params_st {
    size_t *pValue;
    enum cudaLimit limit;
} cudaDeviceGetLimit_v3020_params;

typedef struct cudaDeviceGetCacheConfig_v3020_params_st {
    enum cudaFuncCache *pCacheConfig;
} cudaDeviceGetCacheConfig_v3020_params;

typedef struct cudaDeviceGetStreamPriorityRange_v5050_params_st {
    int *leastPriority;
    int *greatestPriority;
} cudaDeviceGetStreamPriorityRange_v5050_params;

typedef struct cudaDeviceSetCacheConfig_v3020_params_st {
    enum cudaFuncCache cacheConfig;
} cudaDeviceSetCacheConfig_v3020_params;

typedef struct cudaDeviceGetSharedMemConfig_v4020_params_st {
    enum cudaSharedMemConfig *pConfig;
} cudaDeviceGetSharedMemConfig_v4020_params;

typedef struct cudaDeviceSetSharedMemConfig_v4020_params_st {
    enum cudaSharedMemConfig config;
} cudaDeviceSetSharedMemConfig_v4020_params;

typedef struct cudaDeviceGetByPCIBusId_v4010_params_st {
    int *device;
    const char *pciBusId;
} cudaDeviceGetByPCIBusId_v4010_params;

typedef struct cudaDeviceGetPCIBusId_v4010_params_st {
    char *pciBusId;
    int len;
    int device;
} cudaDeviceGetPCIBusId_v4010_params;

typedef struct cudaIpcGetEventHandle_v4010_params_st {
    cudaIpcEventHandle_t *handle;
    cudaEvent_t event;
} cudaIpcGetEventHandle_v4010_params;

typedef struct cudaIpcOpenEventHandle_v4010_params_st {
    cudaEvent_t *event;
    cudaIpcEventHandle_t handle;
} cudaIpcOpenEventHandle_v4010_params;

typedef struct cudaIpcGetMemHandle_v4010_params_st {
    cudaIpcMemHandle_t *handle;
    void *devPtr;
} cudaIpcGetMemHandle_v4010_params;

typedef struct cudaIpcOpenMemHandle_v4010_params_st {
    void **devPtr;
    cudaIpcMemHandle_t handle;
    unsigned int flags;
} cudaIpcOpenMemHandle_v4010_params;

typedef struct cudaIpcCloseMemHandle_v4010_params_st {
    void *devPtr;
} cudaIpcCloseMemHandle_v4010_params;

typedef struct cudaThreadSetLimit_v3020_params_st {
    enum cudaLimit limit;
    size_t value;
} cudaThreadSetLimit_v3020_params;

typedef struct cudaThreadGetLimit_v3020_params_st {
    size_t *pValue;
    enum cudaLimit limit;
} cudaThreadGetLimit_v3020_params;

typedef struct cudaThreadGetCacheConfig_v3020_params_st {
    enum cudaFuncCache *pCacheConfig;
} cudaThreadGetCacheConfig_v3020_params;

typedef struct cudaThreadSetCacheConfig_v3020_params_st {
    enum cudaFuncCache cacheConfig;
} cudaThreadSetCacheConfig_v3020_params;

typedef struct cudaGetErrorName_v6050_params_st {
    cudaError_t error;
} cudaGetErrorName_v6050_params;

typedef struct cudaGetErrorString_v3020_params_st {
    cudaError_t error;
} cudaGetErrorString_v3020_params;

typedef struct cudaGetDeviceCount_v3020_params_st {
    int *count;
} cudaGetDeviceCount_v3020_params;

typedef struct cudaGetDeviceProperties_v3020_params_st {
    struct cudaDeviceProp *prop;
    int device;
} cudaGetDeviceProperties_v3020_params;

typedef struct cudaDeviceGetAttribute_v5000_params_st {
    int *value;
    enum cudaDeviceAttr attr;
    int device;
} cudaDeviceGetAttribute_v5000_params;

typedef struct cudaDeviceGetNvSciSyncAttributes_v10020_params_st {
    void *nvSciSyncAttrList;
    int device;
    int flags;
} cudaDeviceGetNvSciSyncAttributes_v10020_params;

typedef struct cudaDeviceGetP2PAttribute_v8000_params_st {
    int *value;
    enum cudaDeviceP2PAttr attr;
    int srcDevice;
    int dstDevice;
} cudaDeviceGetP2PAttribute_v8000_params;

typedef struct cudaChooseDevice_v3020_params_st {
    int *device;
    const struct cudaDeviceProp *prop;
} cudaChooseDevice_v3020_params;

typedef struct cudaSetDevice_v3020_params_st {
    int device;
} cudaSetDevice_v3020_params;

typedef struct cudaGetDevice_v3020_params_st {
    int *device;
} cudaGetDevice_v3020_params;

typedef struct cudaSetValidDevices_v3020_params_st {
    int *device_arr;
    int len;
} cudaSetValidDevices_v3020_params;

typedef struct cudaSetDeviceFlags_v3020_params_st {
    unsigned int flags;
} cudaSetDeviceFlags_v3020_params;

typedef struct cudaGetDeviceFlags_v7000_params_st {
    unsigned int *flags;
} cudaGetDeviceFlags_v7000_params;

typedef struct cudaStreamCreate_v3020_params_st {
    cudaStream_t *pStream;
} cudaStreamCreate_v3020_params;

typedef struct cudaStreamCreateWithFlags_v5000_params_st {
    cudaStream_t *pStream;
    unsigned int flags;
} cudaStreamCreateWithFlags_v5000_params;

typedef struct cudaStreamCreateWithPriority_v5050_params_st {
    cudaStream_t *pStream;
    unsigned int flags;
    int priority;
} cudaStreamCreateWithPriority_v5050_params;

typedef struct cudaStreamGetPriority_ptsz_v7000_params_st {
    cudaStream_t hStream;
    int *priority;
} cudaStreamGetPriority_ptsz_v7000_params;

typedef struct cudaStreamGetFlags_ptsz_v7000_params_st {
    cudaStream_t hStream;
    unsigned int *flags;
} cudaStreamGetFlags_ptsz_v7000_params;

typedef struct cudaStreamDestroy_v5050_params_st {
    cudaStream_t stream;
} cudaStreamDestroy_v5050_params;

typedef struct cudaStreamWaitEvent_ptsz_v7000_params_st {
    cudaStream_t stream;
    cudaEvent_t event;
    unsigned int flags;
} cudaStreamWaitEvent_ptsz_v7000_params;

typedef struct cudaStreamAddCallback_ptsz_v7000_params_st {
    cudaStream_t stream;
    cudaStreamCallback_t callback;
    void *userData;
    unsigned int flags;
} cudaStreamAddCallback_ptsz_v7000_params;

typedef struct cudaStreamSynchronize_ptsz_v7000_params_st {
    cudaStream_t stream;
} cudaStreamSynchronize_ptsz_v7000_params;

typedef struct cudaStreamQuery_ptsz_v7000_params_st {
    cudaStream_t stream;
} cudaStreamQuery_ptsz_v7000_params;

typedef struct cudaStreamAttachMemAsync_ptsz_v7000_params_st {
    cudaStream_t stream;
    void *devPtr;
    size_t length;
    unsigned int flags;
} cudaStreamAttachMemAsync_ptsz_v7000_params;

typedef struct cudaStreamBeginCapture_ptsz_v10000_params_st {
    cudaStream_t stream;
    enum cudaStreamCaptureMode mode;
} cudaStreamBeginCapture_ptsz_v10000_params;

typedef struct cudaThreadExchangeStreamCaptureMode_v10010_params_st {
    enum cudaStreamCaptureMode *mode;
} cudaThreadExchangeStreamCaptureMode_v10010_params;

typedef struct cudaStreamEndCapture_ptsz_v10000_params_st {
    cudaStream_t stream;
    cudaGraph_t *pGraph;
} cudaStreamEndCapture_ptsz_v10000_params;

typedef struct cudaStreamIsCapturing_ptsz_v10000_params_st {
    cudaStream_t stream;
    enum cudaStreamCaptureStatus *pCaptureStatus;
} cudaStreamIsCapturing_ptsz_v10000_params;

typedef struct cudaStreamGetCaptureInfo_ptsz_v10010_params_st {
    cudaStream_t stream;
    enum cudaStreamCaptureStatus *pCaptureStatus;
    unsigned long long *pId;
} cudaStreamGetCaptureInfo_ptsz_v10010_params;

typedef struct cudaEventCreate_v3020_params_st {
    cudaEvent_t *event;
} cudaEventCreate_v3020_params;

typedef struct cudaEventCreateWithFlags_v3020_params_st {
    cudaEvent_t *event;
    unsigned int flags;
} cudaEventCreateWithFlags_v3020_params;

typedef struct cudaEventRecord_ptsz_v7000_params_st {
    cudaEvent_t event;
    cudaStream_t stream;
} cudaEventRecord_ptsz_v7000_params;

typedef struct cudaEventQuery_v3020_params_st {
    cudaEvent_t event;
} cudaEventQuery_v3020_params;

typedef struct cudaEventSynchronize_v3020_params_st {
    cudaEvent_t event;
} cudaEventSynchronize_v3020_params;

typedef struct cudaEventDestroy_v3020_params_st {
    cudaEvent_t event;
} cudaEventDestroy_v3020_params;

typedef struct cudaEventElapsedTime_v3020_params_st {
    float *ms;
    cudaEvent_t start;
    cudaEvent_t end;
} cudaEventElapsedTime_v3020_params;

typedef struct cudaImportExternalMemory_v10000_params_st {
    cudaExternalMemory_t *extMem_out;
    const struct cudaExternalMemoryHandleDesc *memHandleDesc;
} cudaImportExternalMemory_v10000_params;

typedef struct cudaExternalMemoryGetMappedBuffer_v10000_params_st {
    void **devPtr;
    cudaExternalMemory_t extMem;
    const struct cudaExternalMemoryBufferDesc *bufferDesc;
} cudaExternalMemoryGetMappedBuffer_v10000_params;

typedef struct cudaExternalMemoryGetMappedMipmappedArray_v10000_params_st {
    cudaMipmappedArray_t *mipmap;
    cudaExternalMemory_t extMem;
    const struct cudaExternalMemoryMipmappedArrayDesc *mipmapDesc;
} cudaExternalMemoryGetMappedMipmappedArray_v10000_params;

typedef struct cudaDestroyExternalMemory_v10000_params_st {
    cudaExternalMemory_t extMem;
} cudaDestroyExternalMemory_v10000_params;

typedef struct cudaImportExternalSemaphore_v10000_params_st {
    cudaExternalSemaphore_t *extSem_out;
    const struct cudaExternalSemaphoreHandleDesc *semHandleDesc;
} cudaImportExternalSemaphore_v10000_params;

typedef struct cudaSignalExternalSemaphoresAsync_ptsz_v10000_params_st {
    const cudaExternalSemaphore_t *extSemArray;
    const struct cudaExternalSemaphoreSignalParams *paramsArray;
    unsigned int numExtSems;
    cudaStream_t stream;
} cudaSignalExternalSemaphoresAsync_ptsz_v10000_params;

typedef struct cudaWaitExternalSemaphoresAsync_ptsz_v10000_params_st {
    const cudaExternalSemaphore_t *extSemArray;
    const struct cudaExternalSemaphoreWaitParams *paramsArray;
    unsigned int numExtSems;
    cudaStream_t stream;
} cudaWaitExternalSemaphoresAsync_ptsz_v10000_params;

typedef struct cudaDestroyExternalSemaphore_v10000_params_st {
    cudaExternalSemaphore_t extSem;
} cudaDestroyExternalSemaphore_v10000_params;

typedef struct cudaLaunchKernel_ptsz_v7000_params_st {
    const void *func;
    dim3 gridDim;
    dim3 blockDim;
    void **args;
    size_t sharedMem;
    cudaStream_t stream;
} cudaLaunchKernel_ptsz_v7000_params;

typedef struct cudaLaunchCooperativeKernel_ptsz_v9000_params_st {
    const void *func;
    dim3 gridDim;
    dim3 blockDim;
    void **args;
    size_t sharedMem;
    cudaStream_t stream;
} cudaLaunchCooperativeKernel_ptsz_v9000_params;

typedef struct cudaLaunchCooperativeKernelMultiDevice_v9000_params_st {
    struct cudaLaunchParams *launchParamsList;
    unsigned int numDevices;
    unsigned int flags;
} cudaLaunchCooperativeKernelMultiDevice_v9000_params;

typedef struct cudaFuncSetCacheConfig_v3020_params_st {
    const void *func;
    enum cudaFuncCache cacheConfig;
} cudaFuncSetCacheConfig_v3020_params;

typedef struct cudaFuncSetSharedMemConfig_v4020_params_st {
    const void *func;
    enum cudaSharedMemConfig config;
} cudaFuncSetSharedMemConfig_v4020_params;

typedef struct cudaFuncGetAttributes_v3020_params_st {
    struct cudaFuncAttributes *attr;
    const void *func;
} cudaFuncGetAttributes_v3020_params;

typedef struct cudaFuncSetAttribute_v9000_params_st {
    const void *func;
    enum cudaFuncAttribute attr;
    int value;
} cudaFuncSetAttribute_v9000_params;

typedef struct cudaSetDoubleForDevice_v3020_params_st {
    double *d;
} cudaSetDoubleForDevice_v3020_params;

typedef struct cudaSetDoubleForHost_v3020_params_st {
    double *d;
} cudaSetDoubleForHost_v3020_params;

typedef struct cudaLaunchHostFunc_ptsz_v10000_params_st {
    cudaStream_t stream;
    cudaHostFn_t fn;
    void *userData;
} cudaLaunchHostFunc_ptsz_v10000_params;

typedef struct cudaOccupancyMaxActiveBlocksPerMultiprocessor_v6050_params_st {
    int *numBlocks;
    const void *func;
    int blockSize;
    size_t dynamicSMemSize;
} cudaOccupancyMaxActiveBlocksPerMultiprocessor_v6050_params;

typedef struct cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_v7000_params_st {
    int *numBlocks;
    const void *func;
    int blockSize;
    size_t dynamicSMemSize;
    unsigned int flags;
} cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_v7000_params;

typedef struct cudaMallocManaged_v6000_params_st {
    void **devPtr;
    size_t size;
    unsigned int flags;
} cudaMallocManaged_v6000_params;

typedef struct cudaMalloc_v3020_params_st {
    void **devPtr;
    size_t size;
} cudaMalloc_v3020_params;

typedef struct cudaMallocHost_v3020_params_st {
    void **ptr;
    size_t size;
} cudaMallocHost_v3020_params;

typedef struct cudaMallocPitch_v3020_params_st {
    void **devPtr;
    size_t *pitch;
    size_t width;
    size_t height;
} cudaMallocPitch_v3020_params;

typedef struct cudaMallocArray_v3020_params_st {
    cudaArray_t *array;
    const struct cudaChannelFormatDesc *desc;
    size_t width;
    size_t height;
    unsigned int flags;
} cudaMallocArray_v3020_params;

typedef struct cudaFree_v3020_params_st {
    void *devPtr;
} cudaFree_v3020_params;

typedef struct cudaFreeHost_v3020_params_st {
    void *ptr;
} cudaFreeHost_v3020_params;

typedef struct cudaFreeArray_v3020_params_st {
    cudaArray_t array;
} cudaFreeArray_v3020_params;

typedef struct cudaFreeMipmappedArray_v5000_params_st {
    cudaMipmappedArray_t mipmappedArray;
} cudaFreeMipmappedArray_v5000_params;

typedef struct cudaHostAlloc_v3020_params_st {
    void **pHost;
    size_t size;
    unsigned int flags;
} cudaHostAlloc_v3020_params;

typedef struct cudaHostRegister_v4000_params_st {
    void *ptr;
    size_t size;
    unsigned int flags;
} cudaHostRegister_v4000_params;

typedef struct cudaHostUnregister_v4000_params_st {
    void *ptr;
} cudaHostUnregister_v4000_params;

typedef struct cudaHostGetDevicePointer_v3020_params_st {
    void **pDevice;
    void *pHost;
    unsigned int flags;
} cudaHostGetDevicePointer_v3020_params;

typedef struct cudaHostGetFlags_v3020_params_st {
    unsigned int *pFlags;
    void *pHost;
} cudaHostGetFlags_v3020_params;

typedef struct cudaMalloc3D_v3020_params_st {
    struct cudaPitchedPtr *pitchedDevPtr;
    struct cudaExtent extent;
} cudaMalloc3D_v3020_params;

typedef struct cudaMalloc3DArray_v3020_params_st {
    cudaArray_t *array;
    const struct cudaChannelFormatDesc *desc;
    struct cudaExtent extent;
    unsigned int flags;
} cudaMalloc3DArray_v3020_params;

typedef struct cudaMallocMipmappedArray_v5000_params_st {
    cudaMipmappedArray_t *mipmappedArray;
    const struct cudaChannelFormatDesc *desc;
    struct cudaExtent extent;
    unsigned int numLevels;
    unsigned int flags;
} cudaMallocMipmappedArray_v5000_params;

typedef struct cudaGetMipmappedArrayLevel_v5000_params_st {
    cudaArray_t *levelArray;
    cudaMipmappedArray_const_t mipmappedArray;
    unsigned int level;
} cudaGetMipmappedArrayLevel_v5000_params;

typedef struct cudaMemcpy3D_ptds_v7000_params_st {
    const struct cudaMemcpy3DParms *p;
} cudaMemcpy3D_ptds_v7000_params;

typedef struct cudaMemcpy3DPeer_ptds_v7000_params_st {
    const struct cudaMemcpy3DPeerParms *p;
} cudaMemcpy3DPeer_ptds_v7000_params;

typedef struct cudaMemcpy3DAsync_ptsz_v7000_params_st {
    const struct cudaMemcpy3DParms *p;
    cudaStream_t stream;
} cudaMemcpy3DAsync_ptsz_v7000_params;

typedef struct cudaMemcpy3DPeerAsync_ptsz_v7000_params_st {
    const struct cudaMemcpy3DPeerParms *p;
    cudaStream_t stream;
} cudaMemcpy3DPeerAsync_ptsz_v7000_params;

typedef struct cudaMemGetInfo_v3020_params_st {
    size_t *free;
    size_t *total;
} cudaMemGetInfo_v3020_params;

typedef struct cudaArrayGetInfo_v4010_params_st {
    struct cudaChannelFormatDesc *desc;
    struct cudaExtent *extent;
    unsigned int *flags;
    cudaArray_t array;
} cudaArrayGetInfo_v4010_params;

typedef struct cudaMemcpy_ptds_v7000_params_st {
    void *dst;
    const void *src;
    size_t count;
    enum cudaMemcpyKind kind;
} cudaMemcpy_ptds_v7000_params;

typedef struct cudaMemcpyPeer_v4000_params_st {
    void *dst;
    int dstDevice;
    const void *src;
    int srcDevice;
    size_t count;
} cudaMemcpyPeer_v4000_params;

typedef struct cudaMemcpy2D_ptds_v7000_params_st {
    void *dst;
    size_t dpitch;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
} cudaMemcpy2D_ptds_v7000_params;

typedef struct cudaMemcpy2DToArray_ptds_v7000_params_st {
    cudaArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
} cudaMemcpy2DToArray_ptds_v7000_params;

typedef struct cudaMemcpy2DFromArray_ptds_v7000_params_st {
    void *dst;
    size_t dpitch;
    cudaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
} cudaMemcpy2DFromArray_ptds_v7000_params;

typedef struct cudaMemcpy2DArrayToArray_ptds_v7000_params_st {
    cudaArray_t dst;
    size_t wOffsetDst;
    size_t hOffsetDst;
    cudaArray_const_t src;
    size_t wOffsetSrc;
    size_t hOffsetSrc;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
} cudaMemcpy2DArrayToArray_ptds_v7000_params;

typedef struct cudaMemcpyToSymbol_ptds_v7000_params_st {
    const void *symbol;
    const void *src;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
} cudaMemcpyToSymbol_ptds_v7000_params;

typedef struct cudaMemcpyFromSymbol_ptds_v7000_params_st {
    void *dst;
    const void *symbol;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
} cudaMemcpyFromSymbol_ptds_v7000_params;

typedef struct cudaMemcpyAsync_ptsz_v7000_params_st {
    void *dst;
    const void *src;
    size_t count;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
} cudaMemcpyAsync_ptsz_v7000_params;

typedef struct cudaMemcpyPeerAsync_v4000_params_st {
    void *dst;
    int dstDevice;
    const void *src;
    int srcDevice;
    size_t count;
    cudaStream_t stream;
} cudaMemcpyPeerAsync_v4000_params;

typedef struct cudaMemcpy2DAsync_ptsz_v7000_params_st {
    void *dst;
    size_t dpitch;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
} cudaMemcpy2DAsync_ptsz_v7000_params;

typedef struct cudaMemcpy2DToArrayAsync_ptsz_v7000_params_st {
    cudaArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
} cudaMemcpy2DToArrayAsync_ptsz_v7000_params;

typedef struct cudaMemcpy2DFromArrayAsync_ptsz_v7000_params_st {
    void *dst;
    size_t dpitch;
    cudaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
} cudaMemcpy2DFromArrayAsync_ptsz_v7000_params;

typedef struct cudaMemcpyToSymbolAsync_ptsz_v7000_params_st {
    const void *symbol;
    const void *src;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
} cudaMemcpyToSymbolAsync_ptsz_v7000_params;

typedef struct cudaMemcpyFromSymbolAsync_ptsz_v7000_params_st {
    void *dst;
    const void *symbol;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
} cudaMemcpyFromSymbolAsync_ptsz_v7000_params;

typedef struct cudaMemset_ptds_v7000_params_st {
    void *devPtr;
    int value;
    size_t count;
} cudaMemset_ptds_v7000_params;

typedef struct cudaMemset2D_ptds_v7000_params_st {
    void *devPtr;
    size_t pitch;
    int value;
    size_t width;
    size_t height;
} cudaMemset2D_ptds_v7000_params;

typedef struct cudaMemset3D_ptds_v7000_params_st {
    struct cudaPitchedPtr pitchedDevPtr;
    int value;
    struct cudaExtent extent;
} cudaMemset3D_ptds_v7000_params;

typedef struct cudaMemsetAsync_ptsz_v7000_params_st {
    void *devPtr;
    int value;
    size_t count;
    cudaStream_t stream;
} cudaMemsetAsync_ptsz_v7000_params;

typedef struct cudaMemset2DAsync_ptsz_v7000_params_st {
    void *devPtr;
    size_t pitch;
    int value;
    size_t width;
    size_t height;
    cudaStream_t stream;
} cudaMemset2DAsync_ptsz_v7000_params;

typedef struct cudaMemset3DAsync_ptsz_v7000_params_st {
    struct cudaPitchedPtr pitchedDevPtr;
    int value;
    struct cudaExtent extent;
    cudaStream_t stream;
} cudaMemset3DAsync_ptsz_v7000_params;

typedef struct cudaGetSymbolAddress_v3020_params_st {
    void **devPtr;
    const void *symbol;
} cudaGetSymbolAddress_v3020_params;

typedef struct cudaGetSymbolSize_v3020_params_st {
    size_t *size;
    const void *symbol;
} cudaGetSymbolSize_v3020_params;

typedef struct cudaMemPrefetchAsync_ptsz_v8000_params_st {
    const void *devPtr;
    size_t count;
    int dstDevice;
    cudaStream_t stream;
} cudaMemPrefetchAsync_ptsz_v8000_params;

typedef struct cudaMemAdvise_v8000_params_st {
    const void *devPtr;
    size_t count;
    enum cudaMemoryAdvise advice;
    int device;
} cudaMemAdvise_v8000_params;

typedef struct cudaMemRangeGetAttribute_v8000_params_st {
    void *data;
    size_t dataSize;
    enum cudaMemRangeAttribute attribute;
    const void *devPtr;
    size_t count;
} cudaMemRangeGetAttribute_v8000_params;

typedef struct cudaMemRangeGetAttributes_v8000_params_st {
    void **data;
    size_t *dataSizes;
    enum cudaMemRangeAttribute *attributes;
    size_t numAttributes;
    const void *devPtr;
    size_t count;
} cudaMemRangeGetAttributes_v8000_params;

typedef struct cudaMemcpyToArray_ptds_v7000_params_st {
    cudaArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t count;
    enum cudaMemcpyKind kind;
} cudaMemcpyToArray_ptds_v7000_params;

typedef struct cudaMemcpyFromArray_ptds_v7000_params_st {
    void *dst;
    cudaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t count;
    enum cudaMemcpyKind kind;
} cudaMemcpyFromArray_ptds_v7000_params;

typedef struct cudaMemcpyArrayToArray_ptds_v7000_params_st {
    cudaArray_t dst;
    size_t wOffsetDst;
    size_t hOffsetDst;
    cudaArray_const_t src;
    size_t wOffsetSrc;
    size_t hOffsetSrc;
    size_t count;
    enum cudaMemcpyKind kind;
} cudaMemcpyArrayToArray_ptds_v7000_params;

typedef struct cudaMemcpyToArrayAsync_ptsz_v7000_params_st {
    cudaArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t count;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
} cudaMemcpyToArrayAsync_ptsz_v7000_params;

typedef struct cudaMemcpyFromArrayAsync_ptsz_v7000_params_st {
    void *dst;
    cudaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t count;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
} cudaMemcpyFromArrayAsync_ptsz_v7000_params;

typedef struct cudaPointerGetAttributes_v4000_params_st {
    struct cudaPointerAttributes *attributes;
    const void *ptr;
} cudaPointerGetAttributes_v4000_params;

typedef struct cudaDeviceCanAccessPeer_v4000_params_st {
    int *canAccessPeer;
    int device;
    int peerDevice;
} cudaDeviceCanAccessPeer_v4000_params;

typedef struct cudaDeviceEnablePeerAccess_v4000_params_st {
    int peerDevice;
    unsigned int flags;
} cudaDeviceEnablePeerAccess_v4000_params;

typedef struct cudaDeviceDisablePeerAccess_v4000_params_st {
    int peerDevice;
} cudaDeviceDisablePeerAccess_v4000_params;

typedef struct cudaGraphicsUnregisterResource_v3020_params_st {
    cudaGraphicsResource_t resource;
} cudaGraphicsUnregisterResource_v3020_params;

typedef struct cudaGraphicsResourceSetMapFlags_v3020_params_st {
    cudaGraphicsResource_t resource;
    unsigned int flags;
} cudaGraphicsResourceSetMapFlags_v3020_params;

typedef struct cudaGraphicsMapResources_v3020_params_st {
    int count;
    cudaGraphicsResource_t *resources;
    cudaStream_t stream;
} cudaGraphicsMapResources_v3020_params;

typedef struct cudaGraphicsUnmapResources_v3020_params_st {
    int count;
    cudaGraphicsResource_t *resources;
    cudaStream_t stream;
} cudaGraphicsUnmapResources_v3020_params;

typedef struct cudaGraphicsResourceGetMappedPointer_v3020_params_st {
    void **devPtr;
    size_t *size;
    cudaGraphicsResource_t resource;
} cudaGraphicsResourceGetMappedPointer_v3020_params;

typedef struct cudaGraphicsSubResourceGetMappedArray_v3020_params_st {
    cudaArray_t *array;
    cudaGraphicsResource_t resource;
    unsigned int arrayIndex;
    unsigned int mipLevel;
} cudaGraphicsSubResourceGetMappedArray_v3020_params;

typedef struct cudaGraphicsResourceGetMappedMipmappedArray_v5000_params_st {
    cudaMipmappedArray_t *mipmappedArray;
    cudaGraphicsResource_t resource;
} cudaGraphicsResourceGetMappedMipmappedArray_v5000_params;

typedef struct cudaBindTexture_v3020_params_st {
    size_t *offset;
    const struct textureReference *texref;
    const void *devPtr;
    const struct cudaChannelFormatDesc *desc;
    size_t size;
} cudaBindTexture_v3020_params;

typedef struct cudaBindTexture2D_v3020_params_st {
    size_t *offset;
    const struct textureReference *texref;
    const void *devPtr;
    const struct cudaChannelFormatDesc *desc;
    size_t width;
    size_t height;
    size_t pitch;
} cudaBindTexture2D_v3020_params;

typedef struct cudaBindTextureToArray_v3020_params_st {
    const struct textureReference *texref;
    cudaArray_const_t array;
    const struct cudaChannelFormatDesc *desc;
} cudaBindTextureToArray_v3020_params;

typedef struct cudaBindTextureToMipmappedArray_v5000_params_st {
    const struct textureReference *texref;
    cudaMipmappedArray_const_t mipmappedArray;
    const struct cudaChannelFormatDesc *desc;
} cudaBindTextureToMipmappedArray_v5000_params;

typedef struct cudaUnbindTexture_v3020_params_st {
    const struct textureReference *texref;
} cudaUnbindTexture_v3020_params;

typedef struct cudaGetTextureAlignmentOffset_v3020_params_st {
    size_t *offset;
    const struct textureReference *texref;
} cudaGetTextureAlignmentOffset_v3020_params;

typedef struct cudaGetTextureReference_v3020_params_st {
    const struct textureReference **texref;
    const void *symbol;
} cudaGetTextureReference_v3020_params;

typedef struct cudaBindSurfaceToArray_v3020_params_st {
    const struct surfaceReference *surfref;
    cudaArray_const_t array;
    const struct cudaChannelFormatDesc *desc;
} cudaBindSurfaceToArray_v3020_params;

typedef struct cudaGetSurfaceReference_v3020_params_st {
    const struct surfaceReference **surfref;
    const void *symbol;
} cudaGetSurfaceReference_v3020_params;

typedef struct cudaGetChannelDesc_v3020_params_st {
    struct cudaChannelFormatDesc *desc;
    cudaArray_const_t array;
} cudaGetChannelDesc_v3020_params;

typedef struct cudaCreateChannelDesc_v3020_params_st {
    int x;
    int y;
    int z;
    int w;
    enum cudaChannelFormatKind f;
} cudaCreateChannelDesc_v3020_params;

typedef struct cudaCreateTextureObject_v5000_params_st {
    cudaTextureObject_t *pTexObject;
    const struct cudaResourceDesc *pResDesc;
    const struct cudaTextureDesc *pTexDesc;
    const struct cudaResourceViewDesc *pResViewDesc;
} cudaCreateTextureObject_v5000_params;

typedef struct cudaDestroyTextureObject_v5000_params_st {
    cudaTextureObject_t texObject;
} cudaDestroyTextureObject_v5000_params;

typedef struct cudaGetTextureObjectResourceDesc_v5000_params_st {
    struct cudaResourceDesc *pResDesc;
    cudaTextureObject_t texObject;
} cudaGetTextureObjectResourceDesc_v5000_params;

typedef struct cudaGetTextureObjectTextureDesc_v5000_params_st {
    struct cudaTextureDesc *pTexDesc;
    cudaTextureObject_t texObject;
} cudaGetTextureObjectTextureDesc_v5000_params;

typedef struct cudaGetTextureObjectResourceViewDesc_v5000_params_st {
    struct cudaResourceViewDesc *pResViewDesc;
    cudaTextureObject_t texObject;
} cudaGetTextureObjectResourceViewDesc_v5000_params;

typedef struct cudaCreateSurfaceObject_v5000_params_st {
    cudaSurfaceObject_t *pSurfObject;
    const struct cudaResourceDesc *pResDesc;
} cudaCreateSurfaceObject_v5000_params;

typedef struct cudaDestroySurfaceObject_v5000_params_st {
    cudaSurfaceObject_t surfObject;
} cudaDestroySurfaceObject_v5000_params;

typedef struct cudaGetSurfaceObjectResourceDesc_v5000_params_st {
    struct cudaResourceDesc *pResDesc;
    cudaSurfaceObject_t surfObject;
} cudaGetSurfaceObjectResourceDesc_v5000_params;

typedef struct cudaDriverGetVersion_v3020_params_st {
    int *driverVersion;
} cudaDriverGetVersion_v3020_params;

typedef struct cudaRuntimeGetVersion_v3020_params_st {
    int *runtimeVersion;
} cudaRuntimeGetVersion_v3020_params;

typedef struct cudaGraphCreate_v10000_params_st {
    cudaGraph_t *pGraph;
    unsigned int flags;
} cudaGraphCreate_v10000_params;

typedef struct cudaGraphAddKernelNode_v10000_params_st {
    cudaGraphNode_t *pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t *pDependencies;
    size_t numDependencies;
    const struct cudaKernelNodeParams *pNodeParams;
} cudaGraphAddKernelNode_v10000_params;

typedef struct cudaGraphKernelNodeGetParams_v10000_params_st {
    cudaGraphNode_t node;
    struct cudaKernelNodeParams *pNodeParams;
} cudaGraphKernelNodeGetParams_v10000_params;

typedef struct cudaGraphKernelNodeSetParams_v10000_params_st {
    cudaGraphNode_t node;
    const struct cudaKernelNodeParams *pNodeParams;
} cudaGraphKernelNodeSetParams_v10000_params;

typedef struct cudaGraphAddMemcpyNode_v10000_params_st {
    cudaGraphNode_t *pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t *pDependencies;
    size_t numDependencies;
    const struct cudaMemcpy3DParms *pCopyParams;
} cudaGraphAddMemcpyNode_v10000_params;

typedef struct cudaGraphMemcpyNodeGetParams_v10000_params_st {
    cudaGraphNode_t node;
    struct cudaMemcpy3DParms *pNodeParams;
} cudaGraphMemcpyNodeGetParams_v10000_params;

typedef struct cudaGraphMemcpyNodeSetParams_v10000_params_st {
    cudaGraphNode_t node;
    const struct cudaMemcpy3DParms *pNodeParams;
} cudaGraphMemcpyNodeSetParams_v10000_params;

typedef struct cudaGraphAddMemsetNode_v10000_params_st {
    cudaGraphNode_t *pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t *pDependencies;
    size_t numDependencies;
    const struct cudaMemsetParams *pMemsetParams;
} cudaGraphAddMemsetNode_v10000_params;

typedef struct cudaGraphMemsetNodeGetParams_v10000_params_st {
    cudaGraphNode_t node;
    struct cudaMemsetParams *pNodeParams;
} cudaGraphMemsetNodeGetParams_v10000_params;

typedef struct cudaGraphMemsetNodeSetParams_v10000_params_st {
    cudaGraphNode_t node;
    const struct cudaMemsetParams *pNodeParams;
} cudaGraphMemsetNodeSetParams_v10000_params;

typedef struct cudaGraphAddHostNode_v10000_params_st {
    cudaGraphNode_t *pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t *pDependencies;
    size_t numDependencies;
    const struct cudaHostNodeParams *pNodeParams;
} cudaGraphAddHostNode_v10000_params;

typedef struct cudaGraphHostNodeGetParams_v10000_params_st {
    cudaGraphNode_t node;
    struct cudaHostNodeParams *pNodeParams;
} cudaGraphHostNodeGetParams_v10000_params;

typedef struct cudaGraphHostNodeSetParams_v10000_params_st {
    cudaGraphNode_t node;
    const struct cudaHostNodeParams *pNodeParams;
} cudaGraphHostNodeSetParams_v10000_params;

typedef struct cudaGraphAddChildGraphNode_v10000_params_st {
    cudaGraphNode_t *pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t *pDependencies;
    size_t numDependencies;
    cudaGraph_t childGraph;
} cudaGraphAddChildGraphNode_v10000_params;

typedef struct cudaGraphChildGraphNodeGetGraph_v10000_params_st {
    cudaGraphNode_t node;
    cudaGraph_t *pGraph;
} cudaGraphChildGraphNodeGetGraph_v10000_params;

typedef struct cudaGraphAddEmptyNode_v10000_params_st {
    cudaGraphNode_t *pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t *pDependencies;
    size_t numDependencies;
} cudaGraphAddEmptyNode_v10000_params;

typedef struct cudaGraphClone_v10000_params_st {
    cudaGraph_t *pGraphClone;
    cudaGraph_t originalGraph;
} cudaGraphClone_v10000_params;

typedef struct cudaGraphNodeFindInClone_v10000_params_st {
    cudaGraphNode_t *pNode;
    cudaGraphNode_t originalNode;
    cudaGraph_t clonedGraph;
} cudaGraphNodeFindInClone_v10000_params;

typedef struct cudaGraphNodeGetType_v10000_params_st {
    cudaGraphNode_t node;
    enum cudaGraphNodeType *pType;
} cudaGraphNodeGetType_v10000_params;

typedef struct cudaGraphGetNodes_v10000_params_st {
    cudaGraph_t graph;
    cudaGraphNode_t *nodes;
    size_t *numNodes;
} cudaGraphGetNodes_v10000_params;

typedef struct cudaGraphGetRootNodes_v10000_params_st {
    cudaGraph_t graph;
    cudaGraphNode_t *pRootNodes;
    size_t *pNumRootNodes;
} cudaGraphGetRootNodes_v10000_params;

typedef struct cudaGraphGetEdges_v10000_params_st {
    cudaGraph_t graph;
    cudaGraphNode_t *from;
    cudaGraphNode_t *to;
    size_t *numEdges;
} cudaGraphGetEdges_v10000_params;

typedef struct cudaGraphNodeGetDependencies_v10000_params_st {
    cudaGraphNode_t node;
    cudaGraphNode_t *pDependencies;
    size_t *pNumDependencies;
} cudaGraphNodeGetDependencies_v10000_params;

typedef struct cudaGraphNodeGetDependentNodes_v10000_params_st {
    cudaGraphNode_t node;
    cudaGraphNode_t *pDependentNodes;
    size_t *pNumDependentNodes;
} cudaGraphNodeGetDependentNodes_v10000_params;

typedef struct cudaGraphAddDependencies_v10000_params_st {
    cudaGraph_t graph;
    const cudaGraphNode_t *from;
    const cudaGraphNode_t *to;
    size_t numDependencies;
} cudaGraphAddDependencies_v10000_params;

typedef struct cudaGraphRemoveDependencies_v10000_params_st {
    cudaGraph_t graph;
    const cudaGraphNode_t *from;
    const cudaGraphNode_t *to;
    size_t numDependencies;
} cudaGraphRemoveDependencies_v10000_params;

typedef struct cudaGraphDestroyNode_v10000_params_st {
    cudaGraphNode_t node;
} cudaGraphDestroyNode_v10000_params;

typedef struct cudaGraphInstantiate_v10000_params_st {
    cudaGraphExec_t *pGraphExec;
    cudaGraph_t graph;
    cudaGraphNode_t *pErrorNode;
    char *pLogBuffer;
    size_t bufferSize;
} cudaGraphInstantiate_v10000_params;

typedef struct cudaGraphExecKernelNodeSetParams_v10010_params_st {
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t node;
    const struct cudaKernelNodeParams *pNodeParams;
} cudaGraphExecKernelNodeSetParams_v10010_params;

typedef struct cudaGraphExecMemcpyNodeSetParams_v10020_params_st {
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t node;
    const struct cudaMemcpy3DParms *pNodeParams;
} cudaGraphExecMemcpyNodeSetParams_v10020_params;

typedef struct cudaGraphExecMemsetNodeSetParams_v10020_params_st {
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t node;
    const struct cudaMemsetParams *pNodeParams;
} cudaGraphExecMemsetNodeSetParams_v10020_params;

typedef struct cudaGraphExecHostNodeSetParams_v10020_params_st {
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t node;
    const struct cudaHostNodeParams *pNodeParams;
} cudaGraphExecHostNodeSetParams_v10020_params;

typedef struct cudaGraphExecUpdate_v10020_params_st {
    cudaGraphExec_t hGraphExec;
    cudaGraph_t hGraph;
    cudaGraphNode_t *hErrorNode_out;
    enum cudaGraphExecUpdateResult *updateResult_out;
} cudaGraphExecUpdate_v10020_params;

typedef struct cudaGraphLaunch_ptsz_v10000_params_st {
    cudaGraphExec_t graphExec;
    cudaStream_t stream;
} cudaGraphLaunch_ptsz_v10000_params;

typedef struct cudaGraphExecDestroy_v10000_params_st {
    cudaGraphExec_t graphExec;
} cudaGraphExecDestroy_v10000_params;

typedef struct cudaGraphDestroy_v10000_params_st {
    cudaGraph_t graph;
} cudaGraphDestroy_v10000_params;

typedef struct cudaMemcpy_v3020_params_st {
    void *dst;
    const void *src;
    size_t count;
    enum cudaMemcpyKind kind;
} cudaMemcpy_v3020_params;

typedef struct cudaMemcpyToSymbol_v3020_params_st {
    const void *symbol;
    const void *src;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
} cudaMemcpyToSymbol_v3020_params;

typedef struct cudaMemcpyFromSymbol_v3020_params_st {
    void *dst;
    const void *symbol;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
} cudaMemcpyFromSymbol_v3020_params;

typedef struct cudaMemcpy2D_v3020_params_st {
    void *dst;
    size_t dpitch;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
} cudaMemcpy2D_v3020_params;

typedef struct cudaMemcpyToArray_v3020_params_st {
    cudaArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t count;
    enum cudaMemcpyKind kind;
} cudaMemcpyToArray_v3020_params;

typedef struct cudaMemcpy2DToArray_v3020_params_st {
    cudaArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
} cudaMemcpy2DToArray_v3020_params;

typedef struct cudaMemcpyFromArray_v3020_params_st {
    void *dst;
    cudaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t count;
    enum cudaMemcpyKind kind;
} cudaMemcpyFromArray_v3020_params;

typedef struct cudaMemcpy2DFromArray_v3020_params_st {
    void *dst;
    size_t dpitch;
    cudaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
} cudaMemcpy2DFromArray_v3020_params;

typedef struct cudaMemcpyArrayToArray_v3020_params_st {
    cudaArray_t dst;
    size_t wOffsetDst;
    size_t hOffsetDst;
    cudaArray_const_t src;
    size_t wOffsetSrc;
    size_t hOffsetSrc;
    size_t count;
    enum cudaMemcpyKind kind;
} cudaMemcpyArrayToArray_v3020_params;

typedef struct cudaMemcpy2DArrayToArray_v3020_params_st {
    cudaArray_t dst;
    size_t wOffsetDst;
    size_t hOffsetDst;
    cudaArray_const_t src;
    size_t wOffsetSrc;
    size_t hOffsetSrc;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
} cudaMemcpy2DArrayToArray_v3020_params;

typedef struct cudaMemcpy3D_v3020_params_st {
    const struct cudaMemcpy3DParms *p;
} cudaMemcpy3D_v3020_params;

typedef struct cudaMemcpy3DPeer_v4000_params_st {
    const struct cudaMemcpy3DPeerParms *p;
} cudaMemcpy3DPeer_v4000_params;

typedef struct cudaMemset_v3020_params_st {
    void *devPtr;
    int value;
    size_t count;
} cudaMemset_v3020_params;

typedef struct cudaMemset2D_v3020_params_st {
    void *devPtr;
    size_t pitch;
    int value;
    size_t width;
    size_t height;
} cudaMemset2D_v3020_params;

typedef struct cudaMemset3D_v3020_params_st {
    struct cudaPitchedPtr pitchedDevPtr;
    int value;
    struct cudaExtent extent;
} cudaMemset3D_v3020_params;

typedef struct cudaMemcpyAsync_v3020_params_st {
    void *dst;
    const void *src;
    size_t count;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
} cudaMemcpyAsync_v3020_params;

typedef struct cudaMemcpyToSymbolAsync_v3020_params_st {
    const void *symbol;
    const void *src;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
} cudaMemcpyToSymbolAsync_v3020_params;

typedef struct cudaMemcpyFromSymbolAsync_v3020_params_st {
    void *dst;
    const void *symbol;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
} cudaMemcpyFromSymbolAsync_v3020_params;

typedef struct cudaMemcpy2DAsync_v3020_params_st {
    void *dst;
    size_t dpitch;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
} cudaMemcpy2DAsync_v3020_params;

typedef struct cudaMemcpyToArrayAsync_v3020_params_st {
    cudaArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t count;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
} cudaMemcpyToArrayAsync_v3020_params;

typedef struct cudaMemcpy2DToArrayAsync_v3020_params_st {
    cudaArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
} cudaMemcpy2DToArrayAsync_v3020_params;

typedef struct cudaMemcpyFromArrayAsync_v3020_params_st {
    void *dst;
    cudaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t count;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
} cudaMemcpyFromArrayAsync_v3020_params;

typedef struct cudaMemcpy2DFromArrayAsync_v3020_params_st {
    void *dst;
    size_t dpitch;
    cudaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
} cudaMemcpy2DFromArrayAsync_v3020_params;

typedef struct cudaMemcpy3DAsync_v3020_params_st {
    const struct cudaMemcpy3DParms *p;
    cudaStream_t stream;
} cudaMemcpy3DAsync_v3020_params;

typedef struct cudaMemcpy3DPeerAsync_v4000_params_st {
    const struct cudaMemcpy3DPeerParms *p;
    cudaStream_t stream;
} cudaMemcpy3DPeerAsync_v4000_params;

typedef struct cudaMemsetAsync_v3020_params_st {
    void *devPtr;
    int value;
    size_t count;
    cudaStream_t stream;
} cudaMemsetAsync_v3020_params;

typedef struct cudaMemset2DAsync_v3020_params_st {
    void *devPtr;
    size_t pitch;
    int value;
    size_t width;
    size_t height;
    cudaStream_t stream;
} cudaMemset2DAsync_v3020_params;

typedef struct cudaMemset3DAsync_v3020_params_st {
    struct cudaPitchedPtr pitchedDevPtr;
    int value;
    struct cudaExtent extent;
    cudaStream_t stream;
} cudaMemset3DAsync_v3020_params;

typedef struct cudaStreamQuery_v3020_params_st {
    cudaStream_t stream;
} cudaStreamQuery_v3020_params;

typedef struct cudaStreamGetFlags_v5050_params_st {
    cudaStream_t hStream;
    unsigned int *flags;
} cudaStreamGetFlags_v5050_params;

typedef struct cudaStreamGetPriority_v5050_params_st {
    cudaStream_t hStream;
    int *priority;
} cudaStreamGetPriority_v5050_params;

typedef struct cudaEventRecord_v3020_params_st {
    cudaEvent_t event;
    cudaStream_t stream;
} cudaEventRecord_v3020_params;

typedef struct cudaStreamWaitEvent_v3020_params_st {
    cudaStream_t stream;
    cudaEvent_t event;
    unsigned int flags;
} cudaStreamWaitEvent_v3020_params;

typedef struct cudaStreamAddCallback_v5000_params_st {
    cudaStream_t stream;
    cudaStreamCallback_t callback;
    void *userData;
    unsigned int flags;
} cudaStreamAddCallback_v5000_params;

typedef struct cudaStreamAttachMemAsync_v6000_params_st {
    cudaStream_t stream;
    void *devPtr;
    size_t length;
    unsigned int flags;
} cudaStreamAttachMemAsync_v6000_params;

typedef struct cudaStreamSynchronize_v3020_params_st {
    cudaStream_t stream;
} cudaStreamSynchronize_v3020_params;

typedef struct cudaLaunchKernel_v7000_params_st {
    const void *func;
    dim3 gridDim;
    dim3 blockDim;
    void **args;
    size_t sharedMem;
    cudaStream_t stream;
} cudaLaunchKernel_v7000_params;

typedef struct cudaLaunchCooperativeKernel_v9000_params_st {
    const void *func;
    dim3 gridDim;
    dim3 blockDim;
    void **args;
    size_t sharedMem;
    cudaStream_t stream;
} cudaLaunchCooperativeKernel_v9000_params;

typedef struct cudaLaunchHostFunc_v10000_params_st {
    cudaStream_t stream;
    cudaHostFn_t fn;
    void *userData;
} cudaLaunchHostFunc_v10000_params;

typedef struct cudaMemPrefetchAsync_v8000_params_st {
    const void *devPtr;
    size_t count;
    int dstDevice;
    cudaStream_t stream;
} cudaMemPrefetchAsync_v8000_params;

typedef struct cudaSignalExternalSemaphoresAsync_v10000_params_st {
    const cudaExternalSemaphore_t *extSemArray;
    const struct cudaExternalSemaphoreSignalParams *paramsArray;
    unsigned int numExtSems;
    cudaStream_t stream;
} cudaSignalExternalSemaphoresAsync_v10000_params;

typedef struct cudaWaitExternalSemaphoresAsync_v10000_params_st {
    const cudaExternalSemaphore_t *extSemArray;
    const struct cudaExternalSemaphoreWaitParams *paramsArray;
    unsigned int numExtSems;
    cudaStream_t stream;
} cudaWaitExternalSemaphoresAsync_v10000_params;

typedef struct cudaGraphLaunch_v10000_params_st {
    cudaGraphExec_t graphExec;
    cudaStream_t stream;
} cudaGraphLaunch_v10000_params;

typedef struct cudaStreamBeginCapture_v10000_params_st {
    cudaStream_t stream;
    enum cudaStreamCaptureMode mode;
} cudaStreamBeginCapture_v10000_params;

typedef struct cudaStreamEndCapture_v10000_params_st {
    cudaStream_t stream;
    cudaGraph_t *pGraph;
} cudaStreamEndCapture_v10000_params;

typedef struct cudaStreamIsCapturing_v10000_params_st {
    cudaStream_t stream;
    enum cudaStreamCaptureStatus *pCaptureStatus;
} cudaStreamIsCapturing_v10000_params;

typedef struct cudaStreamGetCaptureInfo_v10010_params_st {
    cudaStream_t hStream;
    enum cudaStreamCaptureStatus *pCaptureStatus;
    unsigned long long *pId;
} cudaStreamGetCaptureInfo_v10010_params;

// Parameter trace structures for removed functions 


// End of parameter trace structures
