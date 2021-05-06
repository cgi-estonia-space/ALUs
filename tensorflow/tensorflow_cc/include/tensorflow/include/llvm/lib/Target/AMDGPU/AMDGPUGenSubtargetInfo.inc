/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Subtarget Enumeration Source Fragment                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


#ifdef GET_SUBTARGETINFO_ENUM
#undef GET_SUBTARGETINFO_ENUM

namespace llvm {
namespace AMDGPU {
enum {
  Feature16BitInsts = 0,
  FeatureAddNoCarryInsts = 1,
  FeatureApertureRegs = 2,
  FeatureAtomicFaddInsts = 3,
  FeatureAutoWaitcntBeforeBarrier = 4,
  FeatureCIInsts = 5,
  FeatureCuMode = 6,
  FeatureDLInsts = 7,
  FeatureDPP = 8,
  FeatureDPP8 = 9,
  FeatureDisable = 10,
  FeatureDoesNotSupportSRAMECC = 11,
  FeatureDoesNotSupportXNACK = 12,
  FeatureDot1Insts = 13,
  FeatureDot2Insts = 14,
  FeatureDot3Insts = 15,
  FeatureDot4Insts = 16,
  FeatureDot5Insts = 17,
  FeatureDot6Insts = 18,
  FeatureDsSrc2Insts = 19,
  FeatureDumpCode = 20,
  FeatureDumpCodeLower = 21,
  FeatureEnableDS128 = 22,
  FeatureEnableLoadStoreOpt = 23,
  FeatureEnablePRTStrictNull = 24,
  FeatureEnableSIScheduler = 25,
  FeatureEnableUnsafeDSOffsetFolding = 26,
  FeatureFMA = 27,
  FeatureFP64 = 28,
  FeatureFastDenormalF32 = 29,
  FeatureFastFMAF32 = 30,
  FeatureFlatAddressSpace = 31,
  FeatureFlatForGlobal = 32,
  FeatureFlatGlobalInsts = 33,
  FeatureFlatInstOffsets = 34,
  FeatureFlatScratchInsts = 35,
  FeatureFlatSegmentOffsetBug = 36,
  FeatureFmaMixInsts = 37,
  FeatureG16 = 38,
  FeatureGCN3Encoding = 39,
  FeatureGFX7GFX8GFX9Insts = 40,
  FeatureGFX8Insts = 41,
  FeatureGFX9 = 42,
  FeatureGFX9Insts = 43,
  FeatureGFX10 = 44,
  FeatureGFX10A16 = 45,
  FeatureGFX10Insts = 46,
  FeatureGFX10_3Insts = 47,
  FeatureGFX10_BEncoding = 48,
  FeatureGetWaveIdInst = 49,
  FeatureImageGather4D16Bug = 50,
  FeatureImageStoreD16Bug = 51,
  FeatureInstFwdPrefetchBug = 52,
  FeatureIntClamp = 53,
  FeatureInv2PiInlineImm = 54,
  FeatureLDSBankCount16 = 55,
  FeatureLDSBankCount32 = 56,
  FeatureLdsBranchVmemWARHazard = 57,
  FeatureLdsMisalignedBug = 58,
  FeatureLocalMemorySize0 = 59,
  FeatureLocalMemorySize32768 = 60,
  FeatureLocalMemorySize65536 = 61,
  FeatureMAIInsts = 62,
  FeatureMFMAInlineLiteralBug = 63,
  FeatureMIMG_R128 = 64,
  FeatureMadMacF32Insts = 65,
  FeatureMadMixInsts = 66,
  FeatureMaxPrivateElementSize4 = 67,
  FeatureMaxPrivateElementSize8 = 68,
  FeatureMaxPrivateElementSize16 = 69,
  FeatureMovrel = 70,
  FeatureNSAEncoding = 71,
  FeatureNSAtoVMEMBug = 72,
  FeatureNoDataDepHazard = 73,
  FeatureNoSdstCMPX = 74,
  FeatureOffset3fBug = 75,
  FeaturePkFmacF16Inst = 76,
  FeaturePromoteAlloca = 77,
  FeatureR128A16 = 78,
  FeatureRegisterBanking = 79,
  FeatureSDWA = 80,
  FeatureSDWAMac = 81,
  FeatureSDWAOmod = 82,
  FeatureSDWAOutModsVOPC = 83,
  FeatureSDWAScalar = 84,
  FeatureSDWASdst = 85,
  FeatureSGPRInitBug = 86,
  FeatureSMEMtoVectorWriteHazard = 87,
  FeatureSMemRealTime = 88,
  FeatureSMemTimeInst = 89,
  FeatureSRAMECC = 90,
  FeatureScalarAtomics = 91,
  FeatureScalarFlatScratchInsts = 92,
  FeatureScalarStores = 93,
  FeatureSeaIslands = 94,
  FeatureSouthernIslands = 95,
  FeatureTrapHandler = 96,
  FeatureTrigReducedRange = 97,
  FeatureUnalignedAccessMode = 98,
  FeatureUnalignedBufferAccess = 99,
  FeatureUnalignedDSAccess = 100,
  FeatureUnalignedScratchAccess = 101,
  FeatureUnpackedD16VMem = 102,
  FeatureVGPRIndexMode = 103,
  FeatureVMEMtoScalarWriteHazard = 104,
  FeatureVOP3Literal = 105,
  FeatureVOP3P = 106,
  FeatureVcmpxExecWARHazard = 107,
  FeatureVcmpxPermlaneHazard = 108,
  FeatureVolcanicIslands = 109,
  FeatureVscnt = 110,
  FeatureWavefrontSize16 = 111,
  FeatureWavefrontSize32 = 112,
  FeatureWavefrontSize64 = 113,
  FeatureXNACK = 114,
  HalfRate64Ops = 115,
  NumSubtargetFeatures = 116
};
} // end namespace AMDGPU
} // end namespace llvm

#endif // GET_SUBTARGETINFO_ENUM


#ifdef GET_SUBTARGETINFO_MC_DESC
#undef GET_SUBTARGETINFO_MC_DESC

namespace llvm {
// Sorted (by key) array of values for CPU features.
extern const llvm::SubtargetFeatureKV AMDGPUFeatureKV[] = {
  { "16-bit-insts", "Has i16/f16 instructions", AMDGPU::Feature16BitInsts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "DumpCode", "Dump MachineInstrs in the CodeEmitter", AMDGPU::FeatureDumpCode, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "a16", "Support gfx10-style A16 for 16-bit coordinates/gradients/lod/clamp/mip image operands", AMDGPU::FeatureGFX10A16, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "add-no-carry-insts", "Have VALU add/sub instructions without carry out", AMDGPU::FeatureAddNoCarryInsts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "aperture-regs", "Has Memory Aperture Base and Size Registers", AMDGPU::FeatureApertureRegs, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "atomic-fadd-insts", "Has buffer_atomic_add_f32, buffer_atomic_pk_add_f16, global_atomic_add_f32, global_atomic_pk_add_f16 instructions", AMDGPU::FeatureAtomicFaddInsts, { { { 0x200000000ULL, 0x0ULL, 0x0ULL, } } } },
  { "auto-waitcnt-before-barrier", "Hardware automatically inserts waitcnt before barrier", AMDGPU::FeatureAutoWaitcntBeforeBarrier, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ci-insts", "Additional instructions for CI+", AMDGPU::FeatureCIInsts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "cumode", "Enable CU wavefront execution mode", AMDGPU::FeatureCuMode, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "dl-insts", "Has v_fmac_f32 and v_xnor_b32 instructions", AMDGPU::FeatureDLInsts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "dot1-insts", "Has v_dot4_i32_i8 and v_dot8_i32_i4 instructions", AMDGPU::FeatureDot1Insts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "dot2-insts", "Has v_dot2_f32_f16, v_dot2_i32_i16, v_dot2_u32_u16, v_dot4_u32_u8, v_dot8_u32_u4 instructions", AMDGPU::FeatureDot2Insts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "dot3-insts", "Has v_dot8c_i32_i4 instruction", AMDGPU::FeatureDot3Insts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "dot4-insts", "Has v_dot2c_i32_i16 instruction", AMDGPU::FeatureDot4Insts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "dot5-insts", "Has v_dot2c_f32_f16 instruction", AMDGPU::FeatureDot5Insts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "dot6-insts", "Has v_dot4c_i32_i8 instruction", AMDGPU::FeatureDot6Insts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "dpp", "Support DPP (Data Parallel Primitives) extension", AMDGPU::FeatureDPP, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "dpp8", "Support DPP8 (Data Parallel Primitives) extension", AMDGPU::FeatureDPP8, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ds-src2-insts", "Has ds_*_src2 instructions", AMDGPU::FeatureDsSrc2Insts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "dumpcode", "Dump MachineInstrs in the CodeEmitter", AMDGPU::FeatureDumpCodeLower, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "enable-ds128", "Use ds_{read|write}_b128", AMDGPU::FeatureEnableDS128, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "enable-prt-strict-null", "Enable zeroing of result registers for sparse texture fetches", AMDGPU::FeatureEnablePRTStrictNull, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "fast-denormal-f32", "Enabling denormals does not cause f32 instructions to run at f64 rates", AMDGPU::FeatureFastDenormalF32, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "fast-fmaf", "Assuming f32 fma is at least as fast as mul + add", AMDGPU::FeatureFastFMAF32, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "flat-address-space", "Support flat address space", AMDGPU::FeatureFlatAddressSpace, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "flat-for-global", "Force to generate flat instruction for global", AMDGPU::FeatureFlatForGlobal, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "flat-global-insts", "Have global_* flat memory instructions", AMDGPU::FeatureFlatGlobalInsts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "flat-inst-offsets", "Flat instructions have immediate offset addressing mode", AMDGPU::FeatureFlatInstOffsets, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "flat-scratch-insts", "Have scratch_* flat memory instructions", AMDGPU::FeatureFlatScratchInsts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "flat-segment-offset-bug", "GFX10 bug, inst_offset ignored in flat segment", AMDGPU::FeatureFlatSegmentOffsetBug, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "fma-mix-insts", "Has v_fma_mix_f32, v_fma_mixlo_f16, v_fma_mixhi_f16 instructions", AMDGPU::FeatureFmaMixInsts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "fmaf", "Enable single precision FMA (not as fast as mul+add, but fused)", AMDGPU::FeatureFMA, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "fp64", "Enable double precision operations", AMDGPU::FeatureFP64, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "g16", "Support G16 for 16-bit gradient image operands", AMDGPU::FeatureG16, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "gcn3-encoding", "Encoding format for VI", AMDGPU::FeatureGCN3Encoding, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "get-wave-id-inst", "Has s_get_waveid_in_workgroup instruction", AMDGPU::FeatureGetWaveIdInst, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "gfx10", "GFX10 GPU generation", AMDGPU::FeatureGFX10, { { { 0x20606a6ef0000b27ULL, 0x461001359641ULL, 0x0ULL, } } } },
  { "gfx10-3-insts", "Additional instructions for GFX10.3", AMDGPU::FeatureGFX10_3Insts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "gfx10-insts", "Additional instructions for GFX10+", AMDGPU::FeatureGFX10Insts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "gfx10_b-encoding", "Encoding format GFX10_B", AMDGPU::FeatureGFX10_BEncoding, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "gfx7-gfx8-gfx9-insts", "Instructions shared in GFX7, GFX8, GFX9", AMDGPU::FeatureGFX7GFX8GFX9Insts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "gfx8-insts", "Additional instructions for GFX8+", AMDGPU::FeatureGFX8Insts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "gfx9", "GFX9 GPU generation", AMDGPU::FeatureGFX9, { { { 0x20600b8ef0080127ULL, 0x204903b354002ULL, 0x0ULL, } } } },
  { "gfx9-insts", "Additional instructions for GFX9+", AMDGPU::FeatureGFX9Insts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "half-rate-64-ops", "Most fp64 instructions are half rate instead of quarter", AMDGPU::HalfRate64Ops, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "image-gather4-d16-bug", "Image Gather4 D16 hardware bug", AMDGPU::FeatureImageGather4D16Bug, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "image-store-d16-bug", "Image Store D16 hardware bug", AMDGPU::FeatureImageStoreD16Bug, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "inst-fwd-prefetch-bug", "S_INST_PREFETCH instruction causes shader to hang", AMDGPU::FeatureInstFwdPrefetchBug, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "int-clamp-insts", "Support clamp for integer destination", AMDGPU::FeatureIntClamp, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "inv-2pi-inline-imm", "Has 1 / (2 * pi) as inline immediate", AMDGPU::FeatureInv2PiInlineImm, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "lds-branch-vmem-war-hazard", "Switching between LDS and VMEM-tex not waiting VM_VSRC=0", AMDGPU::FeatureLdsBranchVmemWARHazard, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "lds-misaligned-bug", "Some GFX10 bug with multi-dword LDS and flat access that is not naturally aligned in WGP mode", AMDGPU::FeatureLdsMisalignedBug, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ldsbankcount16", "The number of LDS banks per compute unit.", AMDGPU::FeatureLDSBankCount16, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ldsbankcount32", "The number of LDS banks per compute unit.", AMDGPU::FeatureLDSBankCount32, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "load-store-opt", "Enable SI load/store optimizer pass", AMDGPU::FeatureEnableLoadStoreOpt, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "localmemorysize0", "The size of local memory in bytes", AMDGPU::FeatureLocalMemorySize0, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "localmemorysize32768", "The size of local memory in bytes", AMDGPU::FeatureLocalMemorySize32768, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "localmemorysize65536", "The size of local memory in bytes", AMDGPU::FeatureLocalMemorySize65536, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "mad-mac-f32-insts", "Has v_mad_f32/v_mac_f32/v_madak_f32/v_madmk_f32 instructions", AMDGPU::FeatureMadMacF32Insts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "mad-mix-insts", "Has v_mad_mix_f32, v_mad_mixlo_f16, v_mad_mixhi_f16 instructions", AMDGPU::FeatureMadMixInsts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "mai-insts", "Has mAI instructions", AMDGPU::FeatureMAIInsts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "max-private-element-size-16", "Maximum private access size may be 16", AMDGPU::FeatureMaxPrivateElementSize16, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "max-private-element-size-4", "Maximum private access size may be 4", AMDGPU::FeatureMaxPrivateElementSize4, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "max-private-element-size-8", "Maximum private access size may be 8", AMDGPU::FeatureMaxPrivateElementSize8, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "mfma-inline-literal-bug", "MFMA cannot use inline literal as SrcC", AMDGPU::FeatureMFMAInlineLiteralBug, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "mimg-r128", "Support 128-bit texture resources", AMDGPU::FeatureMIMG_R128, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "movrel", "Has v_movrel*_b32 instructions", AMDGPU::FeatureMovrel, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "no-data-dep-hazard", "Does not need SW waitstates", AMDGPU::FeatureNoDataDepHazard, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "no-sdst-cmpx", "V_CMPX does not write VCC/SGPR in addition to EXEC", AMDGPU::FeatureNoSdstCMPX, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "no-sram-ecc-support", "Hardware does not support SRAM ECC", AMDGPU::FeatureDoesNotSupportSRAMECC, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "no-xnack-support", "Hardware does not support XNACK", AMDGPU::FeatureDoesNotSupportXNACK, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "nsa-encoding", "Support NSA encoding for image instructions", AMDGPU::FeatureNSAEncoding, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "nsa-to-vmem-bug", "MIMG-NSA followed by VMEM fail if EXEC_LO or EXEC_HI equals zero", AMDGPU::FeatureNSAtoVMEMBug, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "offset-3f-bug", "Branch offset of 3f hardware bug", AMDGPU::FeatureOffset3fBug, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "pk-fmac-f16-inst", "Has v_pk_fmac_f16 instruction", AMDGPU::FeaturePkFmacF16Inst, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "promote-alloca", "Enable promote alloca pass", AMDGPU::FeaturePromoteAlloca, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "r128-a16", "Support gfx9-style A16 for 16-bit coordinates/gradients/lod/clamp/mip image operands, where a16 is aliased with r128", AMDGPU::FeatureR128A16, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "register-banking", "Has register banking", AMDGPU::FeatureRegisterBanking, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "s-memrealtime", "Has s_memrealtime instruction", AMDGPU::FeatureSMemRealTime, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "s-memtime-inst", "Has s_memtime instruction", AMDGPU::FeatureSMemTimeInst, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "scalar-atomics", "Has atomic scalar memory instructions", AMDGPU::FeatureScalarAtomics, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "scalar-flat-scratch-insts", "Have s_scratch_* flat memory instructions", AMDGPU::FeatureScalarFlatScratchInsts, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "scalar-stores", "Has store scalar memory instructions", AMDGPU::FeatureScalarStores, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sdwa", "Support SDWA (Sub-DWORD Addressing) extension", AMDGPU::FeatureSDWA, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sdwa-mav", "Support v_mac_f32/f16 with SDWA (Sub-DWORD Addressing) extension", AMDGPU::FeatureSDWAMac, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sdwa-omod", "Support OMod with SDWA (Sub-DWORD Addressing) extension", AMDGPU::FeatureSDWAOmod, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sdwa-out-mods-vopc", "Support clamp for VOPC with SDWA (Sub-DWORD Addressing) extension", AMDGPU::FeatureSDWAOutModsVOPC, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sdwa-scalar", "Support scalar register with SDWA (Sub-DWORD Addressing) extension", AMDGPU::FeatureSDWAScalar, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sdwa-sdst", "Support scalar dst for VOPC with SDWA (Sub-DWORD Addressing) extension", AMDGPU::FeatureSDWASdst, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sea-islands", "SEA_ISLANDS GPU generation", AMDGPU::FeatureSeaIslands, { { { 0x2000010090080820ULL, 0x2000202000043ULL, 0x0ULL, } } } },
  { "sgpr-init-bug", "VI SGPR initialization bug requiring a fixed SGPR allocation size", AMDGPU::FeatureSGPRInitBug, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "si-scheduler", "Enable SI Machine Scheduler", AMDGPU::FeatureEnableSIScheduler, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "smem-to-vector-write-hazard", "s_load_dword followed by v_cmp page faults", AMDGPU::FeatureSMEMtoVectorWriteHazard, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "southern-islands", "SOUTHERN_ISLANDS GPU generation", AMDGPU::FeatureSouthernIslands, { { { 0x1100000010081800ULL, 0x2000202000043ULL, 0x0ULL, } } } },
  { "sram-ecc", "Enable SRAM ECC", AMDGPU::FeatureSRAMECC, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "trap-handler", "Trap handler support", AMDGPU::FeatureTrapHandler, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "trig-reduced-range", "Requires use of fract on arguments to trig instructions", AMDGPU::FeatureTrigReducedRange, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "unaligned-access-mode", "Support unaligned local and region loads and stores", AMDGPU::FeatureUnalignedAccessMode, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "unaligned-buffer-access", "Support unaligned global loads and stores", AMDGPU::FeatureUnalignedBufferAccess, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "unaligned-ds-access", "Does not requires 16 byte alignment for certain local and region loads and stores", AMDGPU::FeatureUnalignedDSAccess, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "unaligned-scratch-access", "Support unaligned scratch loads and stores", AMDGPU::FeatureUnalignedScratchAccess, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "unpacked-d16-vmem", "Has unpacked d16 vmem instructions", AMDGPU::FeatureUnpackedD16VMem, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "unsafe-ds-offset-folding", "Force using DS instruction immediate offsets on SI", AMDGPU::FeatureEnableUnsafeDSOffsetFolding, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "vcmpx-exec-war-hazard", "V_CMPX WAR hazard on EXEC (V_CMPX issue ONLY)", AMDGPU::FeatureVcmpxExecWARHazard, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "vcmpx-permlane-hazard", "TODO: describe me", AMDGPU::FeatureVcmpxPermlaneHazard, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "vgpr-index-mode", "Has VGPR mode register indexing", AMDGPU::FeatureVGPRIndexMode, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "vmem-to-scalar-write-hazard", "VMEM instruction followed by scalar writing to EXEC mask, M0 or SGPR leads to incorrect execution.", AMDGPU::FeatureVMEMtoScalarWriteHazard, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "volcanic-islands", "VOLCANIC_ISLANDS GPU generation", AMDGPU::FeatureVolcanicIslands, { { { 0x20600380b0080921ULL, 0x20082230b0043ULL, 0x0ULL, } } } },
  { "vop3-literal", "Can use one literal in VOP3", AMDGPU::FeatureVOP3Literal, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "vop3p", "Has VOP3P packed instructions", AMDGPU::FeatureVOP3P, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "vscnt", "Has separate store vscnt counter", AMDGPU::FeatureVscnt, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "wavefrontsize16", "The number of threads per wavefront", AMDGPU::FeatureWavefrontSize16, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "wavefrontsize32", "The number of threads per wavefront", AMDGPU::FeatureWavefrontSize32, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "wavefrontsize64", "The number of threads per wavefront", AMDGPU::FeatureWavefrontSize64, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "xnack", "Enable XNACK support", AMDGPU::FeatureXNACK, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
};

#ifdef DBGFIELD
#error "<target>GenSubtargetInfo.inc requires a DBGFIELD macro"
#endif
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
#define DBGFIELD(x) x,
#else
#define DBGFIELD(x)
#endif

// ===============================================================
// Data tables for the new per-operand machine model.

// {ProcResourceIdx, Cycles}
extern const llvm::MCWriteProcResEntry AMDGPUWriteProcResTable[] = {
  { 0,  0}, // Invalid
  { 4,  1}, // #1
  { 5,  1}, // #2
  { 6,  1}, // #3
  { 3,  1}, // #4
  { 2,  1}, // #5
  { 1,  1}, // #6
  { 5,  2}, // #7
  { 7,  2}, // #8
  { 7,  8}, // #9
  { 7, 16}, // #10
  { 4,  1}, // #11
  { 6,  1}, // #12
  { 4,  1}, // #13
  { 7,  1}, // #14
  { 3,  1}, // #15
  { 4,  1}, // #16
  { 2,  1}, // #17
  { 4,  1}, // #18
  { 4,  2}, // #19
  { 5,  1}, // #20
  { 6,  1}, // #21
  { 4,  2}, // #22
  { 6,  2} // #23
}; // AMDGPUWriteProcResTable

// {Cycles, WriteResourceID}
extern const llvm::MCWriteLatencyEntry AMDGPUWriteLatencyTable[] = {
  { 0,  0}, // Invalid
  { 1,  0}, // #1 WriteSALU_Write32Bit_WriteFloatFMA
  {80,  0}, // #2 WriteVMEM
  { 5,  0}, // #3 WriteLDS_WriteSMEM_Write32Bit_WriteFloatCvt_WriteFloatFMA
  { 4,  0}, // #4 WriteExport_WriteTrans32_WriteFloatCvt_WriteDoubleCvt_WriteQuarterRate32_WriteTrans64
  { 8,  0}, // #5 WriteBranch_WriteDoubleAdd_Write8PassMAI_WriteQuarterRate32
  { 1,  0}, // #6 Write32Bit_WriteFloatFMA
  { 1,  0}, // #7 WriteSALU
  { 2,  0}, // #8 Write64Bit_Write2PassMAI_WriteSALU_WriteDoubleAdd
  {16,  0}, // #9 WriteFloatFMA_WriteDouble_WriteTrans64_Write16PassMAI_WriteExport
  {16,  0}, // #10 WriteFloatFMA_WriteDouble
  { 1,  0}, // #11 WriteSALU
  { 4,  0}, // #12 WriteQuarterRate32_WriteDouble
  { 1,  0}, // #13 WriteSALU
  { 2,  0}, // #14 Write64Bit
  { 2,  0}, // #15 Write64Bit
  {500,  0}, // #16 WriteBarrier
  {320,  0}, // #17 WriteVMEM
  {20,  0}, // #18 WriteLDS_WriteSMEM
  {32,  0}, // #19 WriteBranch
  { 5,  0}, // #20 Write32Bit_WriteFloatFMA
  { 2,  0}, // #21 WriteSALU
  {22,  0}, // #22 WriteDoubleAdd_WriteDoubleCvt
  { 6,  0}, // #23 Write64Bit
  {10,  0}, // #24 WriteTrans32
  {22,  0}, // #25 WriteDouble
  { 2,  0}, // #26 WriteSALU
  { 8,  0}, // #27 WriteQuarterRate32
  { 2,  0}, // #28 WriteSALU
  {24,  0}, // #29 WriteTrans64
  { 6,  0}, // #30 Write64Bit
  { 6,  0}, // #31 Write64Bit
  {2000,  0} // #32 WriteBarrier
}; // AMDGPUWriteLatencyTable

// {UseIdx, WriteResourceID, Cycles}
extern const llvm::MCReadAdvanceEntry AMDGPUReadAdvanceTable[] = {
  {0,  0,  0}, // Invalid
  {0,  0, -4}, // #1
  {0,  0, -2} // #2
}; // AMDGPUReadAdvanceTable

// {Name, NumMicroOps, BeginGroup, EndGroup, WriteProcResIdx,#, WriteLatencyIdx,#, ReadAdvanceIdx,#}
static const llvm::MCSchedClassDesc SIQuarterSpeedModelSchedClasses[] = {
  {DBGFIELD("InvalidSchedClass")  16383, false, false,  0, 0,  0, 0,  0, 0},
  {DBGFIELD("NullALU_WriteSALU")  1, false, false,  1, 1,  1, 1,  0, 0}, // #1
  {DBGFIELD("NullALU_Write32Bit") 1, false, false,  2, 1,  1, 1,  0, 0}, // #2
  {DBGFIELD("NullALU_WriteVMEM")  1, false, false,  3, 1,  2, 1,  0, 0}, // #3
  {DBGFIELD("NullALU_WriteLDS")   1, false, false,  4, 1,  3, 1,  0, 0}, // #4
  {DBGFIELD("NullALU_WriteExport") 1, false, false,  5, 1,  4, 1,  0, 0}, // #5
  {DBGFIELD("NullALU_WriteBranch") 1, false, false,  6, 1,  5, 1,  0, 0}, // #6
  {DBGFIELD("NullALU")            16383, false, false,  0, 0,  0, 0,  0, 0}, // #7
  {DBGFIELD("NullALU_WriteSMEM")  1, false, false,  4, 1,  3, 1,  0, 0}, // #8
  {DBGFIELD("NullALU_Write32Bit_WriteSALU") 2, false, false,  1, 2,  6, 2,  0, 0}, // #9
  {DBGFIELD("NullALU_WriteDoubleAdd") 1, false, false,  2, 1,  5, 1,  0, 0}, // #10
  {DBGFIELD("NullALU_Write64Bit") 1, false, false,  2, 1,  8, 1,  0, 0}, // #11
  {DBGFIELD("NullALU_WriteTrans32") 1, false, false,  2, 1,  4, 1,  0, 0}, // #12
  {DBGFIELD("NullALU_WriteFloatCvt") 1, false, false,  2, 1,  4, 1,  0, 0}, // #13
  {DBGFIELD("NullALU_WriteDoubleCvt") 1, false, false,  2, 1,  4, 1,  0, 0}, // #14
  {DBGFIELD("NullALU_WriteFloatFMA") 1, false, false,  2, 1,  9, 1,  0, 0}, // #15
  {DBGFIELD("NullALU_WriteDouble") 1, false, false,  2, 1,  9, 1,  0, 0}, // #16
  {DBGFIELD("NullALU_WriteFloatFMA_WriteSALU") 2, false, false,  1, 2, 10, 2,  0, 0}, // #17
  {DBGFIELD("NullALU_WriteDouble_WriteSALU") 2, false, false,  1, 2, 10, 2,  0, 0}, // #18
  {DBGFIELD("NullALU_WriteQuarterRate32_WriteSALU") 2, false, false,  1, 2, 12, 2,  0, 0}, // #19
  {DBGFIELD("NullALU_WriteQuarterRate32") 1, false, false,  2, 1,  4, 1,  0, 0}, // #20
  {DBGFIELD("NullALU_WriteTrans64") 1, false, false,  2, 1,  9, 1,  0, 0}, // #21
  {DBGFIELD("NullALU_Write64Bit_Write64Bit") 2, false, false,  7, 1, 14, 2,  0, 0}, // #22
  {DBGFIELD("NullALU_WriteBarrier") 1, false, false,  6, 1, 16, 1,  0, 0}, // #23
  {DBGFIELD("V_ACCVGPR_WRITE_B32") 16382, false, false,  0, 0,  0, 0,  0, 0}, // #24
  {DBGFIELD("V_MFMA_F32_4X4X1F32_V_MFMA_F32_4X4X2BF16_V_MFMA_F32_4X4X4F16_V_MFMA_I32_4X4X4I8_V_MFMA_F32_4X4X1F32_vi_V_MFMA_F32_4X4X2BF16_vi_V_MFMA_F32_4X4X4F16_vi_V_MFMA_I32_4X4X4I8_vi") 1, false, false,  8, 1,  8, 1,  1, 1}, // #25
  {DBGFIELD("V_MFMA_F32_16X16X16F16_V_MFMA_F32_16X16X1F32_V_MFMA_F32_16X16X2BF16_V_MFMA_F32_16X16X4F16_V_MFMA_F32_16X16X4F32_V_MFMA_F32_16X16X8BF16_V_MFMA_I32_16X16X16I8_V_MFMA_I32_16X16X4I8_V_MFMA_F32_16X16X16F16_vi_V_MFMA_F32_16X16X1F32_vi_V_MFMA_F32_16X16X2BF16_vi_V_MFMA_F32_16X16X4F16_vi_V_MFMA_F32_16X16X4F32_vi_V_MFMA_F32_16X16X8BF16_vi_V_MFMA_I32_16X16X16I8_vi_V_MFMA_I32_16X16X4I8_vi") 1, false, false,  9, 1,  5, 1,  1, 1}, // #26
  {DBGFIELD("V_MFMA_F32_32X32X1F32_V_MFMA_F32_32X32X2BF16_V_MFMA_F32_32X32X2F32_V_MFMA_F32_32X32X4BF16_V_MFMA_F32_32X32X4F16_V_MFMA_F32_32X32X8F16_V_MFMA_I32_32X32X4I8_V_MFMA_I32_32X32X8I8_V_MFMA_F32_32X32X1F32_vi_V_MFMA_F32_32X32X2BF16_vi_V_MFMA_F32_32X32X2F32_vi_V_MFMA_F32_32X32X4BF16_vi_V_MFMA_F32_32X32X4F16_vi_V_MFMA_F32_32X32X8F16_vi_V_MFMA_I32_32X32X4I8_vi_V_MFMA_I32_32X32X8I8_vi") 1, false, false, 10, 1,  9, 1,  1, 1}, // #27
  {DBGFIELD("COPY")               16382, false, false,  0, 0,  0, 0,  0, 0}, // #28
  {DBGFIELD("Write64Bit_MIVGPRRead") 1, false, false,  2, 1,  8, 1,  2, 1}, // #29
  {DBGFIELD("Write64Bit_ReadDefault") 1, false, false,  2, 1,  8, 1,  0, 0}, // #30
  {DBGFIELD("Write32Bit")         1, false, false,  2, 1,  1, 1,  0, 0}, // #31
  {DBGFIELD("Write64Bit")         1, false, false,  2, 1,  8, 1,  0, 0}, // #32
  {DBGFIELD("WriteSALU")          1, false, false,  1, 1,  1, 1,  0, 0}, // #33
}; // SIQuarterSpeedModelSchedClasses

// {Name, NumMicroOps, BeginGroup, EndGroup, WriteProcResIdx,#, WriteLatencyIdx,#, ReadAdvanceIdx,#}
static const llvm::MCSchedClassDesc GFX10SpeedModelSchedClasses[] = {
  {DBGFIELD("InvalidSchedClass")  16383, false, false,  0, 0,  0, 0,  0, 0},
  {DBGFIELD("NullALU_WriteSALU")  1, false, false,  1, 2,  8, 1,  0, 0}, // #1
  {DBGFIELD("NullALU_Write32Bit") 1, false, false, 11, 2,  3, 1,  0, 0}, // #2
  {DBGFIELD("NullALU_WriteVMEM")  1, false, false, 13, 2, 17, 1,  0, 0}, // #3
  {DBGFIELD("NullALU_WriteLDS")   1, false, false, 15, 2, 18, 1,  0, 0}, // #4
  {DBGFIELD("NullALU_WriteExport") 1, false, false, 17, 2,  9, 1,  0, 0}, // #5
  {DBGFIELD("NullALU_WriteBranch") 1, false, false,  6, 1, 19, 1,  0, 0}, // #6
  {DBGFIELD("NullALU")            16383, false, false,  0, 0,  0, 0,  0, 0}, // #7
  {DBGFIELD("NullALU_WriteSMEM")  1, false, false, 15, 2, 18, 1,  0, 0}, // #8
  {DBGFIELD("NullALU_Write32Bit_WriteSALU") 2, false, false, 19, 3, 20, 2,  0, 0}, // #9
  {DBGFIELD("NullALU_WriteDoubleAdd") 1, false, false, 11, 2, 22, 1,  0, 0}, // #10
  {DBGFIELD("NullALU_Write64Bit") 1, false, false, 11, 2, 23, 1,  0, 0}, // #11
  {DBGFIELD("NullALU_WriteTrans32") 1, false, false, 11, 2, 24, 1,  0, 0}, // #12
  {DBGFIELD("NullALU_WriteFloatCvt") 1, false, false, 11, 2,  3, 1,  0, 0}, // #13
  {DBGFIELD("NullALU_WriteDoubleCvt") 1, false, false, 11, 2, 22, 1,  0, 0}, // #14
  {DBGFIELD("NullALU_WriteFloatFMA") 1, false, false, 11, 2,  3, 1,  0, 0}, // #15
  {DBGFIELD("NullALU_WriteDouble") 1, false, false, 11, 2, 22, 1,  0, 0}, // #16
  {DBGFIELD("NullALU_WriteFloatFMA_WriteSALU") 2, false, false, 19, 3, 20, 2,  0, 0}, // #17
  {DBGFIELD("NullALU_WriteDouble_WriteSALU") 2, false, false, 19, 3, 25, 2,  0, 0}, // #18
  {DBGFIELD("NullALU_WriteQuarterRate32_WriteSALU") 2, false, false, 19, 3, 27, 2,  0, 0}, // #19
  {DBGFIELD("NullALU_WriteQuarterRate32") 1, false, false, 11, 2,  5, 1,  0, 0}, // #20
  {DBGFIELD("NullALU_WriteTrans64") 1, false, false, 11, 2, 29, 1,  0, 0}, // #21
  {DBGFIELD("NullALU_Write64Bit_Write64Bit") 2, false, false, 22, 2, 30, 2,  0, 0}, // #22
  {DBGFIELD("NullALU_WriteBarrier") 1, false, false,  6, 1, 32, 1,  0, 0}, // #23
  {DBGFIELD("V_ACCVGPR_WRITE_B32") 1, false, false, 11, 2,  3, 1,  0, 0}, // #24
  {DBGFIELD("V_MFMA_F32_4X4X1F32_V_MFMA_F32_4X4X2BF16_V_MFMA_F32_4X4X4F16_V_MFMA_I32_4X4X4I8_V_MFMA_F32_4X4X1F32_vi_V_MFMA_F32_4X4X2BF16_vi_V_MFMA_F32_4X4X4F16_vi_V_MFMA_I32_4X4X4I8_vi") 1, false, false, 11, 2,  3, 1,  0, 0}, // #25
  {DBGFIELD("V_MFMA_F32_16X16X16F16_V_MFMA_F32_16X16X1F32_V_MFMA_F32_16X16X2BF16_V_MFMA_F32_16X16X4F16_V_MFMA_F32_16X16X4F32_V_MFMA_F32_16X16X8BF16_V_MFMA_I32_16X16X16I8_V_MFMA_I32_16X16X4I8_V_MFMA_F32_16X16X16F16_vi_V_MFMA_F32_16X16X1F32_vi_V_MFMA_F32_16X16X2BF16_vi_V_MFMA_F32_16X16X4F16_vi_V_MFMA_F32_16X16X4F32_vi_V_MFMA_F32_16X16X8BF16_vi_V_MFMA_I32_16X16X16I8_vi_V_MFMA_I32_16X16X4I8_vi") 1, false, false, 11, 2,  3, 1,  0, 0}, // #26
  {DBGFIELD("V_MFMA_F32_32X32X1F32_V_MFMA_F32_32X32X2BF16_V_MFMA_F32_32X32X2F32_V_MFMA_F32_32X32X4BF16_V_MFMA_F32_32X32X4F16_V_MFMA_F32_32X32X8F16_V_MFMA_I32_32X32X4I8_V_MFMA_I32_32X32X8I8_V_MFMA_F32_32X32X1F32_vi_V_MFMA_F32_32X32X2BF16_vi_V_MFMA_F32_32X32X2F32_vi_V_MFMA_F32_32X32X4BF16_vi_V_MFMA_F32_32X32X4F16_vi_V_MFMA_F32_32X32X8F16_vi_V_MFMA_I32_32X32X4I8_vi_V_MFMA_I32_32X32X8I8_vi") 1, false, false, 11, 2,  3, 1,  0, 0}, // #27
  {DBGFIELD("COPY")               16382, false, false,  0, 0,  0, 0,  0, 0}, // #28
  {DBGFIELD("Write64Bit_MIVGPRRead") 0, false, false,  0, 0,  0, 0,  0, 0}, // #29
  {DBGFIELD("Write64Bit_ReadDefault") 0, false, false,  0, 0,  0, 0,  0, 0}, // #30
  {DBGFIELD("Write32Bit")         1, false, false, 11, 2,  3, 1,  0, 0}, // #31
  {DBGFIELD("Write64Bit")         1, false, false, 11, 2, 23, 1,  0, 0}, // #32
  {DBGFIELD("WriteSALU")          1, false, false,  1, 2,  8, 1,  0, 0}, // #33
}; // GFX10SpeedModelSchedClasses

// {Name, NumMicroOps, BeginGroup, EndGroup, WriteProcResIdx,#, WriteLatencyIdx,#, ReadAdvanceIdx,#}
static const llvm::MCSchedClassDesc SIFullSpeedModelSchedClasses[] = {
  {DBGFIELD("InvalidSchedClass")  16383, false, false,  0, 0,  0, 0,  0, 0},
  {DBGFIELD("NullALU_WriteSALU")  1, false, false,  1, 1,  1, 1,  0, 0}, // #1
  {DBGFIELD("NullALU_Write32Bit") 1, false, false,  2, 1,  1, 1,  0, 0}, // #2
  {DBGFIELD("NullALU_WriteVMEM")  1, false, false,  3, 1,  2, 1,  0, 0}, // #3
  {DBGFIELD("NullALU_WriteLDS")   1, false, false,  4, 1,  3, 1,  0, 0}, // #4
  {DBGFIELD("NullALU_WriteExport") 1, false, false,  5, 1,  4, 1,  0, 0}, // #5
  {DBGFIELD("NullALU_WriteBranch") 1, false, false,  6, 1,  5, 1,  0, 0}, // #6
  {DBGFIELD("NullALU")            16383, false, false,  0, 0,  0, 0,  0, 0}, // #7
  {DBGFIELD("NullALU_WriteSMEM")  1, false, false,  4, 1,  3, 1,  0, 0}, // #8
  {DBGFIELD("NullALU_Write32Bit_WriteSALU") 2, false, false,  1, 2,  6, 2,  0, 0}, // #9
  {DBGFIELD("NullALU_WriteDoubleAdd") 1, false, false,  2, 1,  8, 1,  0, 0}, // #10
  {DBGFIELD("NullALU_Write64Bit") 1, false, false,  2, 1,  8, 1,  0, 0}, // #11
  {DBGFIELD("NullALU_WriteTrans32") 1, false, false,  2, 1,  4, 1,  0, 0}, // #12
  {DBGFIELD("NullALU_WriteFloatCvt") 1, false, false,  2, 1,  4, 1,  0, 0}, // #13
  {DBGFIELD("NullALU_WriteDoubleCvt") 1, false, false,  2, 1,  4, 1,  0, 0}, // #14
  {DBGFIELD("NullALU_WriteFloatFMA") 1, false, false,  2, 1,  1, 1,  0, 0}, // #15
  {DBGFIELD("NullALU_WriteDouble") 1, false, false,  2, 1,  4, 1,  0, 0}, // #16
  {DBGFIELD("NullALU_WriteFloatFMA_WriteSALU") 2, false, false,  1, 2,  6, 2,  0, 0}, // #17
  {DBGFIELD("NullALU_WriteDouble_WriteSALU") 2, false, false,  1, 2, 12, 2,  0, 0}, // #18
  {DBGFIELD("NullALU_WriteQuarterRate32_WriteSALU") 2, false, false,  1, 2, 12, 2,  0, 0}, // #19
  {DBGFIELD("NullALU_WriteQuarterRate32") 1, false, false,  2, 1,  4, 1,  0, 0}, // #20
  {DBGFIELD("NullALU_WriteTrans64") 1, false, false,  2, 1,  4, 1,  0, 0}, // #21
  {DBGFIELD("NullALU_Write64Bit_Write64Bit") 2, false, false,  7, 1, 14, 2,  0, 0}, // #22
  {DBGFIELD("NullALU_WriteBarrier") 1, false, false,  6, 1, 16, 1,  0, 0}, // #23
  {DBGFIELD("V_ACCVGPR_WRITE_B32") 16382, false, false,  0, 0,  0, 0,  0, 0}, // #24
  {DBGFIELD("V_MFMA_F32_4X4X1F32_V_MFMA_F32_4X4X2BF16_V_MFMA_F32_4X4X4F16_V_MFMA_I32_4X4X4I8_V_MFMA_F32_4X4X1F32_vi_V_MFMA_F32_4X4X2BF16_vi_V_MFMA_F32_4X4X4F16_vi_V_MFMA_I32_4X4X4I8_vi") 1, false, false,  8, 1,  8, 1,  1, 1}, // #25
  {DBGFIELD("V_MFMA_F32_16X16X16F16_V_MFMA_F32_16X16X1F32_V_MFMA_F32_16X16X2BF16_V_MFMA_F32_16X16X4F16_V_MFMA_F32_16X16X4F32_V_MFMA_F32_16X16X8BF16_V_MFMA_I32_16X16X16I8_V_MFMA_I32_16X16X4I8_V_MFMA_F32_16X16X16F16_vi_V_MFMA_F32_16X16X1F32_vi_V_MFMA_F32_16X16X2BF16_vi_V_MFMA_F32_16X16X4F16_vi_V_MFMA_F32_16X16X4F32_vi_V_MFMA_F32_16X16X8BF16_vi_V_MFMA_I32_16X16X16I8_vi_V_MFMA_I32_16X16X4I8_vi") 1, false, false,  9, 1,  5, 1,  1, 1}, // #26
  {DBGFIELD("V_MFMA_F32_32X32X1F32_V_MFMA_F32_32X32X2BF16_V_MFMA_F32_32X32X2F32_V_MFMA_F32_32X32X4BF16_V_MFMA_F32_32X32X4F16_V_MFMA_F32_32X32X8F16_V_MFMA_I32_32X32X4I8_V_MFMA_I32_32X32X8I8_V_MFMA_F32_32X32X1F32_vi_V_MFMA_F32_32X32X2BF16_vi_V_MFMA_F32_32X32X2F32_vi_V_MFMA_F32_32X32X4BF16_vi_V_MFMA_F32_32X32X4F16_vi_V_MFMA_F32_32X32X8F16_vi_V_MFMA_I32_32X32X4I8_vi_V_MFMA_I32_32X32X8I8_vi") 1, false, false, 10, 1,  9, 1,  1, 1}, // #27
  {DBGFIELD("COPY")               16382, false, false,  0, 0,  0, 0,  0, 0}, // #28
  {DBGFIELD("Write64Bit_MIVGPRRead") 1, false, false,  2, 1,  8, 1,  2, 1}, // #29
  {DBGFIELD("Write64Bit_ReadDefault") 1, false, false,  2, 1,  8, 1,  0, 0}, // #30
  {DBGFIELD("Write32Bit")         1, false, false,  2, 1,  1, 1,  0, 0}, // #31
  {DBGFIELD("Write64Bit")         1, false, false,  2, 1,  8, 1,  0, 0}, // #32
  {DBGFIELD("WriteSALU")          1, false, false,  1, 1,  1, 1,  0, 0}, // #33
}; // SIFullSpeedModelSchedClasses

#undef DBGFIELD

static const llvm::MCSchedModel NoSchedModel = {
  MCSchedModel::DefaultIssueWidth,
  MCSchedModel::DefaultMicroOpBufferSize,
  MCSchedModel::DefaultLoopMicroOpBufferSize,
  MCSchedModel::DefaultLoadLatency,
  MCSchedModel::DefaultHighLatency,
  MCSchedModel::DefaultMispredictPenalty,
  false, // PostRAScheduler
  false, // CompleteModel
  0, // Processor ID
  nullptr, nullptr, 0, 0, // No instruction-level machine model.
  nullptr, // No Itinerary
  nullptr // No extra processor descriptor
};

static const unsigned SIQuarterSpeedModelProcResourceSubUnits[] = {
  0,  // Invalid
};

// {Name, NumUnits, SuperIdx, BufferSize, SubUnitsIdxBegin}
static const llvm::MCProcResourceDesc SIQuarterSpeedModelProcResources[] = {
  {"InvalidUnit", 0, 0, 0, 0},
  {"HWBranch",        1, 0, 1, nullptr}, // #1
  {"HWExport",        1, 0, 7, nullptr}, // #2
  {"HWLGKM",          1, 0, 31, nullptr}, // #3
  {"HWSALU",          1, 0, 1, nullptr}, // #4
  {"HWVALU",          1, 0, 1, nullptr}, // #5
  {"HWVMEM",          1, 0, 15, nullptr}, // #6
  {"HWXDL",           1, 0, 0, nullptr}, // #7
};

static const llvm::MCSchedModel SIQuarterSpeedModel = {
  1, // IssueWidth
  1, // MicroOpBufferSize
  MCSchedModel::DefaultLoopMicroOpBufferSize,
  MCSchedModel::DefaultLoadLatency,
  MCSchedModel::DefaultHighLatency,
  20, // MispredictPenalty
  true, // PostRAScheduler
  true, // CompleteModel
  1, // Processor ID
  SIQuarterSpeedModelProcResources,
  SIQuarterSpeedModelSchedClasses,
  8,
  34,
  nullptr, // No Itinerary
  nullptr // No extra processor descriptor
};

static const unsigned GFX10SpeedModelProcResourceSubUnits[] = {
  0,  // Invalid
};

// {Name, NumUnits, SuperIdx, BufferSize, SubUnitsIdxBegin}
static const llvm::MCProcResourceDesc GFX10SpeedModelProcResources[] = {
  {"InvalidUnit", 0, 0, 0, 0},
  {"HWBranch",        1, 0, 1, nullptr}, // #1
  {"HWExport",        1, 0, 7, nullptr}, // #2
  {"HWLGKM",          1, 0, 31, nullptr}, // #3
  {"HWRC",            1, 0, 1, nullptr}, // #4
  {"HWSALU",          1, 0, 1, nullptr}, // #5
  {"HWVALU",          1, 0, 1, nullptr}, // #6
  {"HWVMEM",          1, 0, 15, nullptr}, // #7
};

static const llvm::MCSchedModel GFX10SpeedModel = {
  1, // IssueWidth
  1, // MicroOpBufferSize
  MCSchedModel::DefaultLoopMicroOpBufferSize,
  MCSchedModel::DefaultLoadLatency,
  MCSchedModel::DefaultHighLatency,
  20, // MispredictPenalty
  true, // PostRAScheduler
  true, // CompleteModel
  2, // Processor ID
  GFX10SpeedModelProcResources,
  GFX10SpeedModelSchedClasses,
  8,
  34,
  nullptr, // No Itinerary
  nullptr // No extra processor descriptor
};

static const unsigned SIFullSpeedModelProcResourceSubUnits[] = {
  0,  // Invalid
};

// {Name, NumUnits, SuperIdx, BufferSize, SubUnitsIdxBegin}
static const llvm::MCProcResourceDesc SIFullSpeedModelProcResources[] = {
  {"InvalidUnit", 0, 0, 0, 0},
  {"HWBranch",        1, 0, 1, nullptr}, // #1
  {"HWExport",        1, 0, 7, nullptr}, // #2
  {"HWLGKM",          1, 0, 31, nullptr}, // #3
  {"HWSALU",          1, 0, 1, nullptr}, // #4
  {"HWVALU",          1, 0, 1, nullptr}, // #5
  {"HWVMEM",          1, 0, 15, nullptr}, // #6
  {"HWXDL",           1, 0, 0, nullptr}, // #7
};

static const llvm::MCSchedModel SIFullSpeedModel = {
  1, // IssueWidth
  1, // MicroOpBufferSize
  MCSchedModel::DefaultLoopMicroOpBufferSize,
  MCSchedModel::DefaultLoadLatency,
  MCSchedModel::DefaultHighLatency,
  20, // MispredictPenalty
  true, // PostRAScheduler
  true, // CompleteModel
  3, // Processor ID
  SIFullSpeedModelProcResources,
  SIFullSpeedModelSchedClasses,
  8,
  34,
  nullptr, // No Itinerary
  nullptr // No extra processor descriptor
};

// Sorted (by key) array of values for CPU subtype.
extern const llvm::SubtargetSubTypeKV AMDGPUSubTypeKV[] = {
 { "bonaire", { { { 0x100000000001000ULL, 0x40000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "carrizo", { { { 0x100000040000000ULL, 0xc204000000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "fiji", { { { 0x100000000001000ULL, 0x204000000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "generic", { { { 0x0ULL, 0x2000000000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &NoSchedModel },
 { "generic-hsa", { { { 0x80000000ULL, 0x2000000000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &NoSchedModel },
 { "gfx1010", { { { 0x712101000081080ULL, 0x119003a800982ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &GFX10SpeedModel },
 { "gfx1011", { { { 0x7121010000e7080ULL, 0x119003a800982ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &GFX10SpeedModel },
 { "gfx1012", { { { 0x7121010000e7080ULL, 0x119003a800982ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &GFX10SpeedModel },
 { "gfx1030", { { { 0x101900000067080ULL, 0x1000000000080ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &GFX10SpeedModel },
 { "gfx1031", { { { 0x101900000067080ULL, 0x1000000000080ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &GFX10SpeedModel },
 { "gfx1032", { { { 0x101900000067080ULL, 0x1000000000080ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &GFX10SpeedModel },
 { "gfx600", { { { 0x100000040001000ULL, 0x8000080000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIFullSpeedModel },
 { "gfx601", { { { 0x100000000001000ULL, 0x80000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "gfx602", { { { 0x100000000001000ULL, 0x80000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "gfx700", { { { 0x100000000001000ULL, 0x40000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "gfx701", { { { 0x100000040001000ULL, 0x8000040000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIFullSpeedModel },
 { "gfx702", { { { 0x80000040001000ULL, 0x40000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "gfx703", { { { 0x80000000001000ULL, 0x40000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "gfx704", { { { 0x100000000001000ULL, 0x40000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "gfx705", { { { 0x80000000001000ULL, 0x40000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "gfx801", { { { 0x100000040000000ULL, 0xc204000000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "gfx802", { { { 0x100000000001000ULL, 0x204000400000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "gfx803", { { { 0x100000000001000ULL, 0x204000000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "gfx805", { { { 0x100000000001000ULL, 0x204000400000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "gfx810", { { { 0x8c000000000000ULL, 0x4200000000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "gfx900", { { { 0x104040000001800ULL, 0x4ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "gfx902", { { { 0x104040000000800ULL, 0x4000000000004ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "gfx904", { { { 0x104042000001800ULL, 0x0ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "gfx906", { { { 0x104042000007080ULL, 0x8000000000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "gfx908", { { { 0xc10404200007e088ULL, 0x8000004001000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "gfx909", { { { 0x104040000000000ULL, 0x4000000000004ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "hainan", { { { 0x100000000001000ULL, 0x80000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "hawaii", { { { 0x100000040001000ULL, 0x8000040000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIFullSpeedModel },
 { "iceland", { { { 0x100000000001000ULL, 0x204000400000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "kabini", { { { 0x80000000001000ULL, 0x40000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "kaveri", { { { 0x100000000001000ULL, 0x40000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "mullins", { { { 0x80000000001000ULL, 0x40000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "oland", { { { 0x100000000001000ULL, 0x80000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "pitcairn", { { { 0x100000000001000ULL, 0x80000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "polaris10", { { { 0x100000000001000ULL, 0x204000000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "polaris11", { { { 0x100000000001000ULL, 0x204000000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "stoney", { { { 0x8c000000000000ULL, 0x4200000000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "tahiti", { { { 0x100000040001000ULL, 0x8000080000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIFullSpeedModel },
 { "tonga", { { { 0x100000000001000ULL, 0x204000400000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "tongapro", { { { 0x100000000001000ULL, 0x204000400000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
 { "verde", { { { 0x100000000001000ULL, 0x80000000ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &SIQuarterSpeedModel },
};

namespace AMDGPU_MC {
unsigned resolveVariantSchedClassImpl(unsigned SchedClass,
    const MCInst *MI, unsigned CPUID) {
  switch (SchedClass) {
  case 24: // V_ACCVGPR_WRITE_B32
    if (CPUID == 1) { // SIQuarterSpeedModel
      return 30; // Write64Bit_ReadDefault
    }
    if (CPUID == 3) { // SIFullSpeedModel
      return 30; // Write64Bit_ReadDefault
    }
    break;
  case 28: // COPY
    if (CPUID == 1) { // SIQuarterSpeedModel
      return 33; // WriteSALU
    }
    if (CPUID == 2) { // GFX10SpeedModel
      return 33; // WriteSALU
    }
    if (CPUID == 3) { // SIFullSpeedModel
      return 33; // WriteSALU
    }
    break;
  };
  // Don't know how to resolve this scheduling class.
  return 0;
}
} // end namespace AMDGPU_MC

struct AMDGPUGenMCSubtargetInfo : public MCSubtargetInfo {
  AMDGPUGenMCSubtargetInfo(const Triple &TT,
    StringRef CPU, StringRef TuneCPU, StringRef FS,
    ArrayRef<SubtargetFeatureKV> PF,
    ArrayRef<SubtargetSubTypeKV> PD,
    const MCWriteProcResEntry *WPR,
    const MCWriteLatencyEntry *WL,
    const MCReadAdvanceEntry *RA, const InstrStage *IS,
    const unsigned *OC, const unsigned *FP) :
      MCSubtargetInfo(TT, CPU, TuneCPU, FS, PF, PD,
                      WPR, WL, RA, IS, OC, FP) { }

  unsigned resolveVariantSchedClass(unsigned SchedClass,
      const MCInst *MI, unsigned CPUID) const override {
    return AMDGPU_MC::resolveVariantSchedClassImpl(SchedClass, MI, CPUID); 
  }
};

static inline MCSubtargetInfo *createAMDGPUMCSubtargetInfoImpl(const Triple &TT, StringRef CPU, StringRef TuneCPU, StringRef FS) {
  return new AMDGPUGenMCSubtargetInfo(TT, CPU, TuneCPU, FS, AMDGPUFeatureKV, AMDGPUSubTypeKV, 
                      AMDGPUWriteProcResTable, AMDGPUWriteLatencyTable, AMDGPUReadAdvanceTable, 
                      nullptr, nullptr, nullptr);
}

} // end namespace llvm

#endif // GET_SUBTARGETINFO_MC_DESC


#ifdef GET_SUBTARGETINFO_TARGET_DESC
#undef GET_SUBTARGETINFO_TARGET_DESC

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

// ParseSubtargetFeatures - Parses features string setting specified
// subtarget options.
void llvm::AMDGPUSubtarget::ParseSubtargetFeatures(StringRef CPU, StringRef TuneCPU, StringRef FS) {
  LLVM_DEBUG(dbgs() << "\nFeatures:" << FS);
  LLVM_DEBUG(dbgs() << "\nCPU:" << CPU);
  LLVM_DEBUG(dbgs() << "\nTuneCPU:" << TuneCPU << "\n\n");
  InitMCProcessorInfo(CPU, TuneCPU, FS);
  const FeatureBitset& Bits = getFeatureBits();
  if (Bits[AMDGPU::Feature16BitInsts]) Has16BitInsts = true;
  if (Bits[AMDGPU::FeatureAddNoCarryInsts]) AddNoCarryInsts = true;
  if (Bits[AMDGPU::FeatureApertureRegs]) HasApertureRegs = true;
  if (Bits[AMDGPU::FeatureAtomicFaddInsts]) HasAtomicFaddInsts = true;
  if (Bits[AMDGPU::FeatureAutoWaitcntBeforeBarrier]) AutoWaitcntBeforeBarrier = true;
  if (Bits[AMDGPU::FeatureCIInsts]) CIInsts = true;
  if (Bits[AMDGPU::FeatureCuMode]) EnableCuMode = true;
  if (Bits[AMDGPU::FeatureDLInsts]) HasDLInsts = true;
  if (Bits[AMDGPU::FeatureDPP]) HasDPP = true;
  if (Bits[AMDGPU::FeatureDPP8]) HasDPP8 = true;
  if (Bits[AMDGPU::FeatureDisable]) FeatureDisable = true;
  if (Bits[AMDGPU::FeatureDoesNotSupportSRAMECC]) DoesNotSupportSRAMECC = true;
  if (Bits[AMDGPU::FeatureDoesNotSupportXNACK]) DoesNotSupportXNACK = true;
  if (Bits[AMDGPU::FeatureDot1Insts]) HasDot1Insts = true;
  if (Bits[AMDGPU::FeatureDot2Insts]) HasDot2Insts = true;
  if (Bits[AMDGPU::FeatureDot3Insts]) HasDot3Insts = true;
  if (Bits[AMDGPU::FeatureDot4Insts]) HasDot4Insts = true;
  if (Bits[AMDGPU::FeatureDot5Insts]) HasDot5Insts = true;
  if (Bits[AMDGPU::FeatureDot6Insts]) HasDot6Insts = true;
  if (Bits[AMDGPU::FeatureDsSrc2Insts]) HasDsSrc2Insts = true;
  if (Bits[AMDGPU::FeatureDumpCode]) DumpCode = true;
  if (Bits[AMDGPU::FeatureDumpCodeLower]) DumpCode = true;
  if (Bits[AMDGPU::FeatureEnableDS128]) EnableDS128 = true;
  if (Bits[AMDGPU::FeatureEnableLoadStoreOpt]) EnableLoadStoreOpt = true;
  if (Bits[AMDGPU::FeatureEnablePRTStrictNull]) EnablePRTStrictNull = true;
  if (Bits[AMDGPU::FeatureEnableSIScheduler]) EnableSIScheduler = true;
  if (Bits[AMDGPU::FeatureEnableUnsafeDSOffsetFolding]) EnableUnsafeDSOffsetFolding = true;
  if (Bits[AMDGPU::FeatureFMA]) FMA = true;
  if (Bits[AMDGPU::FeatureFP64]) FP64 = true;
  if (Bits[AMDGPU::FeatureFastDenormalF32]) FastDenormalF32 = true;
  if (Bits[AMDGPU::FeatureFastFMAF32]) FastFMAF32 = true;
  if (Bits[AMDGPU::FeatureFlatAddressSpace]) FlatAddressSpace = true;
  if (Bits[AMDGPU::FeatureFlatForGlobal]) FlatForGlobal = true;
  if (Bits[AMDGPU::FeatureFlatGlobalInsts]) FlatGlobalInsts = true;
  if (Bits[AMDGPU::FeatureFlatInstOffsets]) FlatInstOffsets = true;
  if (Bits[AMDGPU::FeatureFlatScratchInsts]) FlatScratchInsts = true;
  if (Bits[AMDGPU::FeatureFlatSegmentOffsetBug]) HasFlatSegmentOffsetBug = true;
  if (Bits[AMDGPU::FeatureFmaMixInsts]) HasFmaMixInsts = true;
  if (Bits[AMDGPU::FeatureG16]) HasG16 = true;
  if (Bits[AMDGPU::FeatureGCN3Encoding]) GCN3Encoding = true;
  if (Bits[AMDGPU::FeatureGFX7GFX8GFX9Insts]) GFX7GFX8GFX9Insts = true;
  if (Bits[AMDGPU::FeatureGFX8Insts]) GFX8Insts = true;
  if (Bits[AMDGPU::FeatureGFX9] && Gen < GCNSubtarget::GFX9) Gen = GCNSubtarget::GFX9;
  if (Bits[AMDGPU::FeatureGFX9Insts]) GFX9Insts = true;
  if (Bits[AMDGPU::FeatureGFX10] && Gen < GCNSubtarget::GFX10) Gen = GCNSubtarget::GFX10;
  if (Bits[AMDGPU::FeatureGFX10A16]) HasGFX10A16 = true;
  if (Bits[AMDGPU::FeatureGFX10Insts]) GFX10Insts = true;
  if (Bits[AMDGPU::FeatureGFX10_3Insts]) GFX10_3Insts = true;
  if (Bits[AMDGPU::FeatureGFX10_BEncoding]) GFX10_BEncoding = true;
  if (Bits[AMDGPU::FeatureGetWaveIdInst]) HasGetWaveIdInst = true;
  if (Bits[AMDGPU::FeatureImageGather4D16Bug]) HasImageGather4D16Bug = true;
  if (Bits[AMDGPU::FeatureImageStoreD16Bug]) HasImageStoreD16Bug = true;
  if (Bits[AMDGPU::FeatureInstFwdPrefetchBug]) HasInstFwdPrefetchBug = true;
  if (Bits[AMDGPU::FeatureIntClamp]) HasIntClamp = true;
  if (Bits[AMDGPU::FeatureInv2PiInlineImm]) HasInv2PiInlineImm = true;
  if (Bits[AMDGPU::FeatureLDSBankCount16] && LDSBankCount < 16) LDSBankCount = 16;
  if (Bits[AMDGPU::FeatureLDSBankCount32] && LDSBankCount < 32) LDSBankCount = 32;
  if (Bits[AMDGPU::FeatureLdsBranchVmemWARHazard]) HasLdsBranchVmemWARHazard = true;
  if (Bits[AMDGPU::FeatureLdsMisalignedBug]) LDSMisalignedBug = true;
  if (Bits[AMDGPU::FeatureLocalMemorySize0] && LocalMemorySize < 0) LocalMemorySize = 0;
  if (Bits[AMDGPU::FeatureLocalMemorySize32768] && LocalMemorySize < 32768) LocalMemorySize = 32768;
  if (Bits[AMDGPU::FeatureLocalMemorySize65536] && LocalMemorySize < 65536) LocalMemorySize = 65536;
  if (Bits[AMDGPU::FeatureMAIInsts]) HasMAIInsts = true;
  if (Bits[AMDGPU::FeatureMFMAInlineLiteralBug]) HasMFMAInlineLiteralBug = true;
  if (Bits[AMDGPU::FeatureMIMG_R128]) MIMG_R128 = true;
  if (Bits[AMDGPU::FeatureMadMacF32Insts]) HasMadMacF32Insts = true;
  if (Bits[AMDGPU::FeatureMadMixInsts]) HasMadMixInsts = true;
  if (Bits[AMDGPU::FeatureMaxPrivateElementSize4] && MaxPrivateElementSize < 4) MaxPrivateElementSize = 4;
  if (Bits[AMDGPU::FeatureMaxPrivateElementSize8] && MaxPrivateElementSize < 8) MaxPrivateElementSize = 8;
  if (Bits[AMDGPU::FeatureMaxPrivateElementSize16] && MaxPrivateElementSize < 16) MaxPrivateElementSize = 16;
  if (Bits[AMDGPU::FeatureMovrel]) HasMovrel = true;
  if (Bits[AMDGPU::FeatureNSAEncoding]) HasNSAEncoding = true;
  if (Bits[AMDGPU::FeatureNSAtoVMEMBug]) HasNSAtoVMEMBug = true;
  if (Bits[AMDGPU::FeatureNoDataDepHazard]) HasNoDataDepHazard = true;
  if (Bits[AMDGPU::FeatureNoSdstCMPX]) HasNoSdstCMPX = true;
  if (Bits[AMDGPU::FeatureOffset3fBug]) HasOffset3fBug = true;
  if (Bits[AMDGPU::FeaturePkFmacF16Inst]) HasPkFmacF16Inst = true;
  if (Bits[AMDGPU::FeaturePromoteAlloca]) EnablePromoteAlloca = true;
  if (Bits[AMDGPU::FeatureR128A16]) HasR128A16 = true;
  if (Bits[AMDGPU::FeatureRegisterBanking]) HasRegisterBanking = true;
  if (Bits[AMDGPU::FeatureSDWA]) HasSDWA = true;
  if (Bits[AMDGPU::FeatureSDWAMac]) HasSDWAMac = true;
  if (Bits[AMDGPU::FeatureSDWAOmod]) HasSDWAOmod = true;
  if (Bits[AMDGPU::FeatureSDWAOutModsVOPC]) HasSDWAOutModsVOPC = true;
  if (Bits[AMDGPU::FeatureSDWAScalar]) HasSDWAScalar = true;
  if (Bits[AMDGPU::FeatureSDWASdst]) HasSDWASdst = true;
  if (Bits[AMDGPU::FeatureSGPRInitBug]) SGPRInitBug = true;
  if (Bits[AMDGPU::FeatureSMEMtoVectorWriteHazard]) HasSMEMtoVectorWriteHazard = true;
  if (Bits[AMDGPU::FeatureSMemRealTime]) HasSMemRealTime = true;
  if (Bits[AMDGPU::FeatureSMemTimeInst]) HasSMemTimeInst = true;
  if (Bits[AMDGPU::FeatureSRAMECC]) EnableSRAMECC = true;
  if (Bits[AMDGPU::FeatureScalarAtomics]) HasScalarAtomics = true;
  if (Bits[AMDGPU::FeatureScalarFlatScratchInsts]) ScalarFlatScratchInsts = true;
  if (Bits[AMDGPU::FeatureScalarStores]) HasScalarStores = true;
  if (Bits[AMDGPU::FeatureSeaIslands] && Gen < GCNSubtarget::SEA_ISLANDS) Gen = GCNSubtarget::SEA_ISLANDS;
  if (Bits[AMDGPU::FeatureSouthernIslands] && Gen < GCNSubtarget::SOUTHERN_ISLANDS) Gen = GCNSubtarget::SOUTHERN_ISLANDS;
  if (Bits[AMDGPU::FeatureTrapHandler]) TrapHandler = true;
  if (Bits[AMDGPU::FeatureTrigReducedRange]) HasTrigReducedRange = true;
  if (Bits[AMDGPU::FeatureUnalignedAccessMode]) UnalignedAccessMode = true;
  if (Bits[AMDGPU::FeatureUnalignedBufferAccess]) UnalignedBufferAccess = true;
  if (Bits[AMDGPU::FeatureUnalignedDSAccess]) UnalignedDSAccess = true;
  if (Bits[AMDGPU::FeatureUnalignedScratchAccess]) UnalignedScratchAccess = true;
  if (Bits[AMDGPU::FeatureUnpackedD16VMem]) HasUnpackedD16VMem = true;
  if (Bits[AMDGPU::FeatureVGPRIndexMode]) HasVGPRIndexMode = true;
  if (Bits[AMDGPU::FeatureVMEMtoScalarWriteHazard]) HasVMEMtoScalarWriteHazard = true;
  if (Bits[AMDGPU::FeatureVOP3Literal]) HasVOP3Literal = true;
  if (Bits[AMDGPU::FeatureVOP3P]) HasVOP3PInsts = true;
  if (Bits[AMDGPU::FeatureVcmpxExecWARHazard]) HasVcmpxExecWARHazard = true;
  if (Bits[AMDGPU::FeatureVcmpxPermlaneHazard]) HasVcmpxPermlaneHazard = true;
  if (Bits[AMDGPU::FeatureVolcanicIslands] && Gen < GCNSubtarget::VOLCANIC_ISLANDS) Gen = GCNSubtarget::VOLCANIC_ISLANDS;
  if (Bits[AMDGPU::FeatureVscnt]) HasVscnt = true;
  if (Bits[AMDGPU::FeatureWavefrontSize16] && WavefrontSizeLog2 < 4) WavefrontSizeLog2 = 4;
  if (Bits[AMDGPU::FeatureWavefrontSize32] && WavefrontSizeLog2 < 5) WavefrontSizeLog2 = 5;
  if (Bits[AMDGPU::FeatureWavefrontSize64] && WavefrontSizeLog2 < 6) WavefrontSizeLog2 = 6;
  if (Bits[AMDGPU::FeatureXNACK]) EnableXNACK = true;
  if (Bits[AMDGPU::HalfRate64Ops]) HalfRate64Ops = true;
}
#endif // GET_SUBTARGETINFO_TARGET_DESC


#ifdef GET_SUBTARGETINFO_HEADER
#undef GET_SUBTARGETINFO_HEADER

namespace llvm {
class DFAPacketizer;
namespace AMDGPU_MC {
unsigned resolveVariantSchedClassImpl(unsigned SchedClass, const MCInst *MI, unsigned CPUID);
} // end namespace AMDGPU_MC

struct AMDGPUGenSubtargetInfo : public TargetSubtargetInfo {
  explicit AMDGPUGenSubtargetInfo(const Triple &TT, StringRef CPU, StringRef TuneCPU, StringRef FS);
public:
  unsigned resolveSchedClass(unsigned SchedClass,  const MachineInstr *DefMI, const TargetSchedModel *SchedModel) const override;
  unsigned resolveVariantSchedClass(unsigned SchedClass, const MCInst *MI, unsigned CPUID) const override;
  DFAPacketizer *createDFAPacketizer(const InstrItineraryData *IID) const;
};
} // end namespace llvm

#endif // GET_SUBTARGETINFO_HEADER


#ifdef GET_SUBTARGETINFO_CTOR
#undef GET_SUBTARGETINFO_CTOR

#include "llvm/CodeGen/TargetSchedule.h"

namespace llvm {
extern const llvm::SubtargetFeatureKV AMDGPUFeatureKV[];
extern const llvm::SubtargetSubTypeKV AMDGPUSubTypeKV[];
extern const llvm::MCWriteProcResEntry AMDGPUWriteProcResTable[];
extern const llvm::MCWriteLatencyEntry AMDGPUWriteLatencyTable[];
extern const llvm::MCReadAdvanceEntry AMDGPUReadAdvanceTable[];
AMDGPUGenSubtargetInfo::AMDGPUGenSubtargetInfo(const Triple &TT, StringRef CPU, StringRef TuneCPU, StringRef FS)
  : TargetSubtargetInfo(TT, CPU, TuneCPU, FS, makeArrayRef(AMDGPUFeatureKV, 115), makeArrayRef(AMDGPUSubTypeKV, 46), 
                        AMDGPUWriteProcResTable, AMDGPUWriteLatencyTable, AMDGPUReadAdvanceTable, 
                        nullptr, nullptr, nullptr) {}

unsigned AMDGPUGenSubtargetInfo
::resolveSchedClass(unsigned SchedClass, const MachineInstr *MI, const TargetSchedModel *SchedModel) const {

  const SIInstrInfo *TII =
    static_cast<const SIInstrInfo*>(SchedModel->getInstrInfo());
  (void)TII;

  switch (SchedClass) {
  case 24: // V_ACCVGPR_WRITE_B32
    if (SchedModel->getProcessorID() == 1) { // SIQuarterSpeedModel
      if (TII->hasVGPRUses(*MI))
        return 29; // Write64Bit_MIVGPRRead
      return 30; // Write64Bit_ReadDefault
    }
    if (SchedModel->getProcessorID() == 3) { // SIFullSpeedModel
      if (TII->hasVGPRUses(*MI))
        return 29; // Write64Bit_MIVGPRRead
      return 30; // Write64Bit_ReadDefault
    }
    break;
  case 28: // COPY
    if (SchedModel->getProcessorID() == 1) { // SIQuarterSpeedModel
      if (TII->isVGPRCopy(*MI) && TII->getOpSize(*MI, 0) <= 32)
        return 31; // Write32Bit
      if (TII->isVGPRCopy(*MI) && TII->getOpSize(*MI, 0) > 32)
        return 32; // Write64Bit
      return 33; // WriteSALU
    }
    if (SchedModel->getProcessorID() == 2) { // GFX10SpeedModel
      if (TII->isVGPRCopy(*MI) && TII->getOpSize(*MI, 0) <= 32)
        return 31; // Write32Bit
      if (TII->isVGPRCopy(*MI) && TII->getOpSize(*MI, 0) > 32)
        return 32; // Write64Bit
      return 33; // WriteSALU
    }
    if (SchedModel->getProcessorID() == 3) { // SIFullSpeedModel
      if (TII->isVGPRCopy(*MI) && TII->getOpSize(*MI, 0) <= 32)
        return 31; // Write32Bit
      if (TII->isVGPRCopy(*MI) && TII->getOpSize(*MI, 0) > 32)
        return 32; // Write64Bit
      return 33; // WriteSALU
    }
    break;
  };
  report_fatal_error("Expected a variant SchedClass");
} // AMDGPUGenSubtargetInfo::resolveSchedClass

unsigned AMDGPUGenSubtargetInfo
::resolveVariantSchedClass(unsigned SchedClass, const MCInst *MI, unsigned CPUID) const {
  return AMDGPU_MC::resolveVariantSchedClassImpl(SchedClass, MI, CPUID);
} // AMDGPUGenSubtargetInfo::resolveVariantSchedClass

} // end namespace llvm

#endif // GET_SUBTARGETINFO_CTOR


#ifdef GET_STIPREDICATE_DECLS_FOR_MC_ANALYSIS
#undef GET_STIPREDICATE_DECLS_FOR_MC_ANALYSIS

#endif // GET_STIPREDICATE_DECLS_FOR_MC_ANALYSIS


#ifdef GET_STIPREDICATE_DEFS_FOR_MC_ANALYSIS
#undef GET_STIPREDICATE_DEFS_FOR_MC_ANALYSIS

#endif // GET_STIPREDICATE_DEFS_FOR_MC_ANALYSIS

