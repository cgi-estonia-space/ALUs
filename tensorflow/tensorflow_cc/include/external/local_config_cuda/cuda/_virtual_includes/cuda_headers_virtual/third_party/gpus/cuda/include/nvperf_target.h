#ifndef NVPERF_TARGET_H
#define NVPERF_TARGET_H

/*
 * Copyright 2014-2019  NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.
 *
 * This software and the information contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and conditions
 * of a form of NVIDIA software license agreement.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) && defined(NVPA_SHARED_LIB)
    #pragma GCC visibility push(default)
    #if !defined(NVPW_LOCAL)
        #define NVPW_LOCAL __attribute__ ((visibility ("hidden")))
    #endif
#else
    #if !defined(NVPW_LOCAL)
        #define NVPW_LOCAL
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @file   nvperf_target.h
 */

/***************************************************************************//**
 *  @name   Common Types
 *  @{
 */

#ifndef NVPERF_NVPA_STATUS_DEFINED
#define NVPERF_NVPA_STATUS_DEFINED

    /// Error codes.
    typedef enum NVPA_Status
    {
        /// Success
        NVPA_STATUS_SUCCESS = 0,
        /// Generic error.
        NVPA_STATUS_ERROR = 1,
        /// Internal error.  Please file a bug!
        NVPA_STATUS_INTERNAL_ERROR = 2,
        /// NVPA_Init() has not been called yet.
        NVPA_STATUS_NOT_INITIALIZED = 3,
        /// The NvPerfAPI DLL/DSO could not be loaded during init.
        NVPA_STATUS_NOT_LOADED = 4,
        /// The function was not found in this version of the NvPerfAPI DLL/DSO.
        NVPA_STATUS_FUNCTION_NOT_FOUND = 5,
        /// The request is intentionally not supported by NvPerfAPI.
        NVPA_STATUS_NOT_SUPPORTED = 6,
        /// The request is not implemented by this version of NvPerfAPI.
        NVPA_STATUS_NOT_IMPLEMENTED = 7,
        /// Invalid argument.
        NVPA_STATUS_INVALID_ARGUMENT = 8,
        /// A MetricId argument does not belong to the specified NVPA_Activity or NVPA_Config.
        NVPA_STATUS_INVALID_METRIC_ID = 9,
        /// No driver has been loaded via NVPA_*_LoadDriver().
        NVPA_STATUS_DRIVER_NOT_LOADED = 10,
        /// Failed memory allocation.
        NVPA_STATUS_OUT_OF_MEMORY = 11,
        /// The request could not be fulfilled due to the state of the current thread.
        NVPA_STATUS_INVALID_THREAD_STATE = 12,
        /// Allocation of context object failed.
        NVPA_STATUS_FAILED_CONTEXT_ALLOC = 13,
        /// The specified GPU is not supported.
        NVPA_STATUS_UNSUPPORTED_GPU = 14,
        /// The installed NVIDIA driver is too old.
        NVPA_STATUS_INSUFFICIENT_DRIVER_VERSION = 15,
        /// Graphics object has not been registered via NVPA_Register*().
        NVPA_STATUS_OBJECT_NOT_REGISTERED = 16,
        /// The operation failed due to a security check.
        NVPA_STATUS_INSUFFICIENT_PRIVILEGE = 17,
        /// The request could not be fulfilled due to the state of the context.
        NVPA_STATUS_INVALID_CONTEXT_STATE = 18,
        /// The request could not be fulfilled due to the state of the object.
        NVPA_STATUS_INVALID_OBJECT_STATE = 19,
        /// The request could not be fulfilled because a system resource is already in use.
        NVPA_STATUS_RESOURCE_UNAVAILABLE = 20,
        /// The NVPA_*_LoadDriver() is called after the context, command queue or device is created.
        NVPA_STATUS_DRIVER_LOADED_TOO_LATE = 21,
        /// The provided buffer is not large enough.
        NVPA_STATUS_INSUFFICIENT_SPACE = 22,
        /// The API object passed to NVPA_[API]_BeginPass/NVPA_[API]_EndPass and
        /// NVPA_[API]_PushRange/NVPA_[API]_PopRange does not match with the NVPA_[API]_BeginSession.
        NVPA_STATUS_OBJECT_MISMATCH = 23,
        NVPA_STATUS__COUNT
    } NVPA_Status;


#endif // NVPERF_NVPA_STATUS_DEFINED


#ifndef NVPERF_NVPA_ACTIVITY_KIND_DEFINED
#define NVPERF_NVPA_ACTIVITY_KIND_DEFINED

    /// The configuration's activity-kind dictates which types of data may be collected.
    typedef enum NVPA_ActivityKind
    {
        /// Invalid value.
        NVPA_ACTIVITY_KIND_INVALID = 0,
        /// A workload-centric activity for serialized and pipelined collection.
        /// 
        /// Profiler is capable of collecting both serialized and pipelined metrics.  The library introduces any
        /// synchronization required to collect serialized metrics.
        NVPA_ACTIVITY_KIND_PROFILER,
        /// A realtime activity for sampling counters from the CPU or GPU.
        NVPA_ACTIVITY_KIND_REALTIME_SAMPLED,
        /// A realtime activity for profiling counters from the CPU or GPU without CPU/GPU synchronizations.
        NVPA_ACTIVITY_KIND_REALTIME_PROFILER,
        NVPA_ACTIVITY_KIND__COUNT
    } NVPA_ActivityKind;


#endif // NVPERF_NVPA_ACTIVITY_KIND_DEFINED


#ifndef NVPERF_NVPA_BOOL_DEFINED
#define NVPERF_NVPA_BOOL_DEFINED
    /// The type used for boolean values.
    typedef uint8_t NVPA_Bool;
#endif // NVPERF_NVPA_BOOL_DEFINED

#ifndef NVPA_STRUCT_SIZE
#define NVPA_STRUCT_SIZE(type_, lastfield_)                     (offsetof(type_, lastfield_) + sizeof(((type_*)0)->lastfield_))
#endif // NVPA_STRUCT_SIZE


#ifndef NVPERF_NVPA_GETPROCADDRESS_DEFINED
#define NVPERF_NVPA_GETPROCADDRESS_DEFINED

typedef NVPA_Status (*NVPA_GenericFn)(void);


    /// 
    /// Gets the address of a PerfWorks API function.
    /// 
    /// \return A function pointer to the function, or NULL if the function is not available.
    /// 
    /// \param pFunctionName [in] Name of the function to retrieve.
    NVPA_GenericFn NVPA_GetProcAddress(const char* pFunctionName);

#endif

#ifndef NVPERF_NVPW_SETLIBRARYLOADPATHS_DEFINED
#define NVPERF_NVPW_SETLIBRARYLOADPATHS_DEFINED


    typedef struct NVPW_SetLibraryLoadPaths_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] number of paths in ppPaths
        size_t numPaths;
        /// [in] array of null-terminated paths
        const char** ppPaths;
    } NVPW_SetLibraryLoadPaths_Params;
#define NVPW_SetLibraryLoadPaths_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_SetLibraryLoadPaths_Params, ppPaths)

    /// Sets library search path for \ref NVPA_InitializeHost() and \ref NVPA_InitializeTarget().
    /// \ref NVPA_InitializeHost() and \ref NVPA_InitializeTarget load the PerfWorks DLL/DSO.  This function sets
    /// ordered paths that will be searched with the LoadLibrary() or dlopen() call.
    /// If load paths are set by this function, the default set of load paths
    /// will not be attempted.
    /// Each path must point at a directory (not a file name).
    /// This function is not thread-safe.
    /// Example Usage:
    /// \code
    ///     const char* paths[] = {
    ///         "path1", "path2", etc
    ///     };
    ///     NVPW_SetLibraryLoadPaths_Params params{NVPW_SetLibraryLoadPaths_Params_STRUCT_SIZE};
    ///     params.numPaths = sizeof(paths)/sizeof(paths[0]);
    ///     params.ppPaths = paths;
    ///     NVPW_SetLibraryLoadPaths(&params);
    ///     NVPA_InitializeHost();
    ///     params.numPaths = 0;
    ///     params.ppPaths = NULL;
    ///     NVPW_SetLibraryLoadPaths(&params);
    /// \endcode
    NVPA_Status NVPW_SetLibraryLoadPaths(NVPW_SetLibraryLoadPaths_Params* pParams);

    typedef struct NVPW_SetLibraryLoadPathsW_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] number of paths in ppwPaths
        size_t numPaths;
        /// [in] array of null-terminated paths
        const wchar_t** ppwPaths;
    } NVPW_SetLibraryLoadPathsW_Params;
#define NVPW_SetLibraryLoadPathsW_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_SetLibraryLoadPathsW_Params, ppwPaths)

    /// Sets library search path for \ref NVPA_InitializeHost() and \ref NVPA_InitializeTarget().
    /// \ref NVPA_InitializeHost() and \ref NVPA_InitializeTarget load the PerfWorks DLL/DSO.  This function sets
    /// ordered paths that will be searched with the LoadLibrary() or dlopen() call.
    /// If load paths are set by this function, the default set of load paths
    /// will not be attempted.
    /// Each path must point at a directory (not a file name).
    /// This function is not thread-safe.
    /// Example Usage:
    /// \code
    ///     const wchar_t* wpaths[] = {
    ///         L"path1", L"path2", etc
    ///     };
    ///     NVPW_SetLibraryLoadPathsW_Params params{NVPW_SetLibraryLoadPathsW_Params_STRUCT_SIZE};
    ///     params.numPaths = sizeof(wpaths)/sizeof(wpaths[0]);
    ///     params.ppwPaths = wpaths;
    ///     NVPW_SetLibraryLoadPathsW(&params);
    ///     NVPA_InitializeHost();
    ///     params.numPaths = 0;
    ///     params.ppwPaths = NULL;
    ///     NVPW_SetLibraryLoadPathsW(&params);
    /// \endcode
    NVPA_Status NVPW_SetLibraryLoadPathsW(NVPW_SetLibraryLoadPathsW_Params* pParams);

#endif

/**
 *  @}
 ******************************************************************************/
 
    /// \deprecated Use of this function is discouraged. Prefer \ref NVPW_InitializeTarget instead.
    NVPA_Status NVPA_InitializeTarget(void);


    // Device enumeration functions must be preceded by NVPA_<API>_LoadDriver(); any API is fine.


    /// \deprecated Use of this function is discouraged. Prefer \ref NVPW_GetDeviceCount instead.
    NVPA_Status NVPA_GetDeviceCount(size_t* pNumDevices);

    /// \deprecated Use of this function is discouraged. Prefer \ref NVPW_Device_GetNames instead.
    NVPA_Status NVPA_Device_GetNames(
        size_t deviceIndex,
        const char** ppDeviceName,
        const char** ppChipName);

    /// \deprecated Use of this function is discouraged. Prefer \ref NVPW_CounterData_GetNumRanges instead.
    NVPA_Status NVPA_CounterData_GetNumRanges(
        const uint8_t* pCounterDataImage,
        size_t* pNumRanges);

    /// \deprecated Use of this function is discouraged. Prefer \ref NVPW_CounterData_GetRangeDescriptions instead.
    NVPA_Status NVPA_CounterData_GetRangeDescriptions(
        const uint8_t* pCounterDataImage,
        size_t rangeIndex,
        size_t numDescriptions,
        const char** ppDescriptions,
        size_t* pNumDescriptions);

#ifndef NVPW_GPU_ARCHITECTURE_SUPPORT_LEVEL_DEFINED
#define NVPW_GPU_ARCHITECTURE_SUPPORT_LEVEL_DEFINED
    typedef enum NVPW_GpuArchitectureSupportLevel
    {
        NVPW_GPU_ARCHITECTURE_SUPPORT_LEVEL_UNKNOWN = 0,
        NVPW_GPU_ARCHITECTURE_SUPPORT_LEVEL_UNSUPPORTED,
        NVPW_GPU_ARCHITECTURE_SUPPORT_LEVEL_SUPPORTED
    } NVPW_GpuArchitectureSupportLevel;
#endif //NVPW_GPU_ARCHITECTURE_SUPPORT_LEVEL_DEFINED

#ifndef NVPW_SLI_SUPPORT_LEVEL_DEFINED
#define NVPW_SLI_SUPPORT_LEVEL_DEFINED
    typedef enum NVPW_SliSupportLevel
    {
        NVPW_SLI_SUPPORT_LEVEL_UNKNOWN = 0,
        NVPW_SLI_SUPPORT_LEVEL_UNSUPPORTED,
        /// Only Non-SLI configurations are supported.
        NVPW_SLI_SUPPORT_LEVEL_SUPPORTED_NON_SLI_CONFIGURATION
    } NVPW_SliSupportLevel;
#endif //NVPW_SLI_SUPPORT_LEVEL_DEFINED


    #define NVPW_FIELD_EXISTS(pParams_, name_) \
        ((pParams_)->structSize >= (const size_t)((const uint8_t*)(&(pParams_)->name_) + sizeof(pParams_)->name_ - (const uint8_t*)(pParams_)))
    

    typedef struct NVPW_InitializeTarget_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
    } NVPW_InitializeTarget_Params;
#define NVPW_InitializeTarget_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_InitializeTarget_Params, pPriv)

    /// Load the target library.
    NVPA_Status NVPW_InitializeTarget(NVPW_InitializeTarget_Params* pParams);

    typedef struct NVPW_GetDeviceCount_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        size_t numDevices;
    } NVPW_GetDeviceCount_Params;
#define NVPW_GetDeviceCount_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_GetDeviceCount_Params, numDevices)

    NVPA_Status NVPW_GetDeviceCount(NVPW_GetDeviceCount_Params* pParams);

    typedef struct NVPW_Device_GetNames_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        size_t deviceIndex;
        const char* pDeviceName;
        const char* pChipName;
    } NVPW_Device_GetNames_Params;
#define NVPW_Device_GetNames_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_Device_GetNames_Params, pChipName)

    NVPA_Status NVPW_Device_GetNames(NVPW_Device_GetNames_Params* pParams);

    typedef struct NVPW_Adapter_GetDeviceIndex_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        struct IDXGIAdapter* pAdapter;
        /// [in]
        size_t sliIndex;
        /// [out]
        size_t deviceIndex;
    } NVPW_Adapter_GetDeviceIndex_Params;
#define NVPW_Adapter_GetDeviceIndex_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_Adapter_GetDeviceIndex_Params, deviceIndex)

    NVPA_Status NVPW_Adapter_GetDeviceIndex(NVPW_Adapter_GetDeviceIndex_Params* pParams);

    typedef struct NVPW_CounterData_GetNumRanges_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        const uint8_t* pCounterDataImage;
        size_t numRanges;
    } NVPW_CounterData_GetNumRanges_Params;
#define NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_CounterData_GetNumRanges_Params, numRanges)

    NVPA_Status NVPW_CounterData_GetNumRanges(NVPW_CounterData_GetNumRanges_Params* pParams);

    typedef struct NVPW_Config_GetNumPasses_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        const uint8_t* pConfig;
        /// [out]
        size_t numPipelinedPasses;
        /// [out]
        size_t numIsolatedPasses;
    } NVPW_Config_GetNumPasses_Params;
#define NVPW_Config_GetNumPasses_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_Config_GetNumPasses_Params, numIsolatedPasses)

    /// Total num passes = numPipelinedPasses + numIsolatedPasses * numNestingLevels
    NVPA_Status NVPW_Config_GetNumPasses(NVPW_Config_GetNumPasses_Params* pParams);

    typedef struct NVPW_CounterData_GetRangeDescriptions_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        const uint8_t* pCounterDataImage;
        size_t rangeIndex;
        /// [inout] Number of descriptions allocated in ppDescriptions
        size_t numDescriptions;
        const char** ppDescriptions;
    } NVPW_CounterData_GetRangeDescriptions_Params;
#define NVPW_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_CounterData_GetRangeDescriptions_Params, ppDescriptions)

    NVPA_Status NVPW_CounterData_GetRangeDescriptions(NVPW_CounterData_GetRangeDescriptions_Params* pParams);

    typedef struct NVPW_Profiler_CounterData_GetRangeDescriptions_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        const uint8_t* pCounterDataImage;
        size_t rangeIndex;
        /// [inout] Number of descriptions allocated in ppDescriptions
        size_t numDescriptions;
        const char** ppDescriptions;
    } NVPW_Profiler_CounterData_GetRangeDescriptions_Params;
#define NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_Profiler_CounterData_GetRangeDescriptions_Params, ppDescriptions)

    NVPA_Status NVPW_Profiler_CounterData_GetRangeDescriptions(NVPW_Profiler_CounterData_GetRangeDescriptions_Params* pParams);

    typedef struct NVPW_PeriodicSampler_CounterData_DelimiterInfo
    {
        const char* pDelimiterName;
        /// defines a half-open interval [rangeIndexStart, rangeIndexEnd)
        uint32_t rangeIndexStart;
        uint32_t rangeIndexEnd;
    } NVPW_PeriodicSampler_CounterData_DelimiterInfo;
#define NVPW_PeriodicSampler_CounterData_DelimiterInfo_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_PeriodicSampler_CounterData_DelimiterInfo, rangeIndexEnd)

    typedef struct NVPW_PeriodicSampler_CounterData_GetDelimiters_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        const uint8_t* pCounterDataImage;
        /// [in]
        size_t delimiterInfoStructSize;
        /// [inout] if pDelimiters is NULL, then the number of delimiters available is returned in numDelimiters,
        /// otherwise numDelimiters should be set by the user to the number of elements in the pDelimiters array, and on
        /// return the variable is overwritten with the number of elements actually written to pDelimiters
        size_t numDelimiters;
        /// [inout] either NULL or a pointer to an array of NVPW_Sampler_CounterData_DelimiterInfo
        NVPW_PeriodicSampler_CounterData_DelimiterInfo* pDelimiters;
    } NVPW_PeriodicSampler_CounterData_GetDelimiters_Params;
#define NVPW_PeriodicSampler_CounterData_GetDelimiters_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_PeriodicSampler_CounterData_GetDelimiters_Params, pDelimiters)

    NVPA_Status NVPW_PeriodicSampler_CounterData_GetDelimiters(NVPW_PeriodicSampler_CounterData_GetDelimiters_Params* pParams);

    typedef struct NVPW_PeriodicSampler_CounterData_GetSampleTime_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        const uint8_t* pCounterDataImage;
        /// [in]
        size_t rangeIndex;
        /// [out]
        uint64_t timestampStart;
        /// [out]
        uint64_t timestampEnd;
    } NVPW_PeriodicSampler_CounterData_GetSampleTime_Params;
#define NVPW_PeriodicSampler_CounterData_GetSampleTime_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_PeriodicSampler_CounterData_GetSampleTime_Params, timestampEnd)

    NVPA_Status NVPW_PeriodicSampler_CounterData_GetSampleTime(NVPW_PeriodicSampler_CounterData_GetSampleTime_Params* pParams);

    typedef struct NVPW_PeriodicSampler_CounterData_TrimInPlace_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        uint8_t* pCounterDataImage;
        /// [in]
        size_t counterDataImageSize;
        /// [out]
        size_t counterDataImageTrimmedSize;
    } NVPW_PeriodicSampler_CounterData_TrimInPlace_Params;
#define NVPW_PeriodicSampler_CounterData_TrimInPlace_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_PeriodicSampler_CounterData_TrimInPlace_Params, counterDataImageTrimmedSize)

    NVPA_Status NVPW_PeriodicSampler_CounterData_TrimInPlace(NVPW_PeriodicSampler_CounterData_TrimInPlace_Params* pParams);



#ifdef __cplusplus
} // extern "C"
#endif

#if defined(__GNUC__) && defined(NVPA_SHARED_LIB)
    #pragma GCC visibility pop
#endif

#endif // NVPERF_TARGET_H
