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
namespace NVPTX {
enum {
  PTX32 = 0,
  PTX40 = 1,
  PTX41 = 2,
  PTX42 = 3,
  PTX43 = 4,
  PTX50 = 5,
  PTX60 = 6,
  PTX61 = 7,
  PTX63 = 8,
  PTX64 = 9,
  PTX65 = 10,
  PTX70 = 11,
  SM20 = 12,
  SM21 = 13,
  SM30 = 14,
  SM32 = 15,
  SM35 = 16,
  SM37 = 17,
  SM50 = 18,
  SM52 = 19,
  SM53 = 20,
  SM60 = 21,
  SM61 = 22,
  SM62 = 23,
  SM70 = 24,
  SM72 = 25,
  SM75 = 26,
  SM80 = 27,
  NumSubtargetFeatures = 28
};
} // end namespace NVPTX
} // end namespace llvm

#endif // GET_SUBTARGETINFO_ENUM


#ifdef GET_SUBTARGETINFO_MC_DESC
#undef GET_SUBTARGETINFO_MC_DESC

namespace llvm {
// Sorted (by key) array of values for CPU features.
extern const llvm::SubtargetFeatureKV NVPTXFeatureKV[] = {
  { "ptx32", "Use PTX version 3.2", NVPTX::PTX32, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ptx40", "Use PTX version 4.0", NVPTX::PTX40, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ptx41", "Use PTX version 4.1", NVPTX::PTX41, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ptx42", "Use PTX version 4.2", NVPTX::PTX42, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ptx43", "Use PTX version 4.3", NVPTX::PTX43, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ptx50", "Use PTX version 5.0", NVPTX::PTX50, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ptx60", "Use PTX version 6.0", NVPTX::PTX60, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ptx61", "Use PTX version 6.1", NVPTX::PTX61, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ptx63", "Use PTX version 6.3", NVPTX::PTX63, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ptx64", "Use PTX version 6.4", NVPTX::PTX64, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ptx65", "Use PTX version 6.5", NVPTX::PTX65, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ptx70", "Use PTX version 7.0", NVPTX::PTX70, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sm_20", "Target SM 2.0", NVPTX::SM20, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sm_21", "Target SM 2.1", NVPTX::SM21, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sm_30", "Target SM 3.0", NVPTX::SM30, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sm_32", "Target SM 3.2", NVPTX::SM32, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sm_35", "Target SM 3.5", NVPTX::SM35, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sm_37", "Target SM 3.7", NVPTX::SM37, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sm_50", "Target SM 5.0", NVPTX::SM50, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sm_52", "Target SM 5.2", NVPTX::SM52, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sm_53", "Target SM 5.3", NVPTX::SM53, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sm_60", "Target SM 6.0", NVPTX::SM60, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sm_61", "Target SM 6.1", NVPTX::SM61, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sm_62", "Target SM 6.2", NVPTX::SM62, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sm_70", "Target SM 7.0", NVPTX::SM70, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sm_72", "Target SM 7.2", NVPTX::SM72, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sm_75", "Target SM 7.5", NVPTX::SM75, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "sm_80", "Target SM 8.0", NVPTX::SM80, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } } },
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
extern const llvm::MCWriteProcResEntry NVPTXWriteProcResTable[] = {
  { 0,  0}, // Invalid
}; // NVPTXWriteProcResTable

// {Cycles, WriteResourceID}
extern const llvm::MCWriteLatencyEntry NVPTXWriteLatencyTable[] = {
  { 0,  0}, // Invalid
}; // NVPTXWriteLatencyTable

// {UseIdx, WriteResourceID, Cycles}
extern const llvm::MCReadAdvanceEntry NVPTXReadAdvanceTable[] = {
  {0,  0,  0}, // Invalid
}; // NVPTXReadAdvanceTable

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

// Sorted (by key) array of values for CPU subtype.
extern const llvm::SubtargetSubTypeKV NVPTXSubTypeKV[] = {
 { "sm_20", { { { 0x1000ULL, 0x0ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &NoSchedModel },
 { "sm_21", { { { 0x2000ULL, 0x0ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &NoSchedModel },
 { "sm_30", { { { 0x4000ULL, 0x0ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &NoSchedModel },
 { "sm_32", { { { 0x8002ULL, 0x0ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &NoSchedModel },
 { "sm_35", { { { 0x10000ULL, 0x0ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &NoSchedModel },
 { "sm_37", { { { 0x20004ULL, 0x0ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &NoSchedModel },
 { "sm_50", { { { 0x40002ULL, 0x0ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &NoSchedModel },
 { "sm_52", { { { 0x80004ULL, 0x0ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &NoSchedModel },
 { "sm_53", { { { 0x100008ULL, 0x0ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &NoSchedModel },
 { "sm_60", { { { 0x200020ULL, 0x0ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &NoSchedModel },
 { "sm_61", { { { 0x400020ULL, 0x0ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &NoSchedModel },
 { "sm_62", { { { 0x800020ULL, 0x0ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &NoSchedModel },
 { "sm_70", { { { 0x1000040ULL, 0x0ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &NoSchedModel },
 { "sm_72", { { { 0x2000080ULL, 0x0ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &NoSchedModel },
 { "sm_75", { { { 0x4000100ULL, 0x0ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &NoSchedModel },
 { "sm_80", { { { 0x8000800ULL, 0x0ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, } } }, &NoSchedModel },
};

namespace NVPTX_MC {
unsigned resolveVariantSchedClassImpl(unsigned SchedClass,
    const MCInst *MI, unsigned CPUID) {
  // Don't know how to resolve this scheduling class.
  return 0;
}
} // end namespace NVPTX_MC

struct NVPTXGenMCSubtargetInfo : public MCSubtargetInfo {
  NVPTXGenMCSubtargetInfo(const Triple &TT,
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
    return NVPTX_MC::resolveVariantSchedClassImpl(SchedClass, MI, CPUID); 
  }
};

static inline MCSubtargetInfo *createNVPTXMCSubtargetInfoImpl(const Triple &TT, StringRef CPU, StringRef TuneCPU, StringRef FS) {
  return new NVPTXGenMCSubtargetInfo(TT, CPU, TuneCPU, FS, NVPTXFeatureKV, NVPTXSubTypeKV, 
                      NVPTXWriteProcResTable, NVPTXWriteLatencyTable, NVPTXReadAdvanceTable, 
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
void llvm::NVPTXSubtarget::ParseSubtargetFeatures(StringRef CPU, StringRef TuneCPU, StringRef FS) {
  LLVM_DEBUG(dbgs() << "\nFeatures:" << FS);
  LLVM_DEBUG(dbgs() << "\nCPU:" << CPU);
  LLVM_DEBUG(dbgs() << "\nTuneCPU:" << TuneCPU << "\n\n");
  InitMCProcessorInfo(CPU, TuneCPU, FS);
  const FeatureBitset& Bits = getFeatureBits();
  if (Bits[NVPTX::PTX32] && PTXVersion < 32) PTXVersion = 32;
  if (Bits[NVPTX::PTX40] && PTXVersion < 40) PTXVersion = 40;
  if (Bits[NVPTX::PTX41] && PTXVersion < 41) PTXVersion = 41;
  if (Bits[NVPTX::PTX42] && PTXVersion < 42) PTXVersion = 42;
  if (Bits[NVPTX::PTX43] && PTXVersion < 43) PTXVersion = 43;
  if (Bits[NVPTX::PTX50] && PTXVersion < 50) PTXVersion = 50;
  if (Bits[NVPTX::PTX60] && PTXVersion < 60) PTXVersion = 60;
  if (Bits[NVPTX::PTX61] && PTXVersion < 61) PTXVersion = 61;
  if (Bits[NVPTX::PTX63] && PTXVersion < 63) PTXVersion = 63;
  if (Bits[NVPTX::PTX64] && PTXVersion < 64) PTXVersion = 64;
  if (Bits[NVPTX::PTX65] && PTXVersion < 65) PTXVersion = 65;
  if (Bits[NVPTX::PTX70] && PTXVersion < 70) PTXVersion = 70;
  if (Bits[NVPTX::SM20] && SmVersion < 20) SmVersion = 20;
  if (Bits[NVPTX::SM21] && SmVersion < 21) SmVersion = 21;
  if (Bits[NVPTX::SM30] && SmVersion < 30) SmVersion = 30;
  if (Bits[NVPTX::SM32] && SmVersion < 32) SmVersion = 32;
  if (Bits[NVPTX::SM35] && SmVersion < 35) SmVersion = 35;
  if (Bits[NVPTX::SM37] && SmVersion < 37) SmVersion = 37;
  if (Bits[NVPTX::SM50] && SmVersion < 50) SmVersion = 50;
  if (Bits[NVPTX::SM52] && SmVersion < 52) SmVersion = 52;
  if (Bits[NVPTX::SM53] && SmVersion < 53) SmVersion = 53;
  if (Bits[NVPTX::SM60] && SmVersion < 60) SmVersion = 60;
  if (Bits[NVPTX::SM61] && SmVersion < 61) SmVersion = 61;
  if (Bits[NVPTX::SM62] && SmVersion < 62) SmVersion = 62;
  if (Bits[NVPTX::SM70] && SmVersion < 70) SmVersion = 70;
  if (Bits[NVPTX::SM72] && SmVersion < 72) SmVersion = 72;
  if (Bits[NVPTX::SM75] && SmVersion < 75) SmVersion = 75;
  if (Bits[NVPTX::SM80] && SmVersion < 80) SmVersion = 80;
}
#endif // GET_SUBTARGETINFO_TARGET_DESC


#ifdef GET_SUBTARGETINFO_HEADER
#undef GET_SUBTARGETINFO_HEADER

namespace llvm {
class DFAPacketizer;
namespace NVPTX_MC {
unsigned resolveVariantSchedClassImpl(unsigned SchedClass, const MCInst *MI, unsigned CPUID);
} // end namespace NVPTX_MC

struct NVPTXGenSubtargetInfo : public TargetSubtargetInfo {
  explicit NVPTXGenSubtargetInfo(const Triple &TT, StringRef CPU, StringRef TuneCPU, StringRef FS);
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
extern const llvm::SubtargetFeatureKV NVPTXFeatureKV[];
extern const llvm::SubtargetSubTypeKV NVPTXSubTypeKV[];
extern const llvm::MCWriteProcResEntry NVPTXWriteProcResTable[];
extern const llvm::MCWriteLatencyEntry NVPTXWriteLatencyTable[];
extern const llvm::MCReadAdvanceEntry NVPTXReadAdvanceTable[];
NVPTXGenSubtargetInfo::NVPTXGenSubtargetInfo(const Triple &TT, StringRef CPU, StringRef TuneCPU, StringRef FS)
  : TargetSubtargetInfo(TT, CPU, TuneCPU, FS, makeArrayRef(NVPTXFeatureKV, 28), makeArrayRef(NVPTXSubTypeKV, 16), 
                        NVPTXWriteProcResTable, NVPTXWriteLatencyTable, NVPTXReadAdvanceTable, 
                        nullptr, nullptr, nullptr) {}

unsigned NVPTXGenSubtargetInfo
::resolveSchedClass(unsigned SchedClass, const MachineInstr *MI, const TargetSchedModel *SchedModel) const {
  report_fatal_error("Expected a variant SchedClass");
} // NVPTXGenSubtargetInfo::resolveSchedClass

unsigned NVPTXGenSubtargetInfo
::resolveVariantSchedClass(unsigned SchedClass, const MCInst *MI, unsigned CPUID) const {
  return NVPTX_MC::resolveVariantSchedClassImpl(SchedClass, MI, CPUID);
} // NVPTXGenSubtargetInfo::resolveVariantSchedClass

} // end namespace llvm

#endif // GET_SUBTARGETINFO_CTOR


#ifdef GET_STIPREDICATE_DECLS_FOR_MC_ANALYSIS
#undef GET_STIPREDICATE_DECLS_FOR_MC_ANALYSIS

#endif // GET_STIPREDICATE_DECLS_FOR_MC_ANALYSIS


#ifdef GET_STIPREDICATE_DEFS_FOR_MC_ANALYSIS
#undef GET_STIPREDICATE_DEFS_FOR_MC_ANALYSIS

#endif // GET_STIPREDICATE_DEFS_FOR_MC_ANALYSIS

