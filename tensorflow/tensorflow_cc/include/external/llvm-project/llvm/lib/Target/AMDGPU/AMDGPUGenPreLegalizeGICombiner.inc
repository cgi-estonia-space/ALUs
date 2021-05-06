/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Global Combiner                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef AMDGPUPRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS
#include "llvm/ADT/SparseBitVector.h"
namespace llvm {
extern cl::OptionCategory GICombinerOptionCategory;
} // end namespace llvm
#endif // ifdef AMDGPUPRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS

#ifdef AMDGPUPRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H
class AMDGPUGenPreLegalizerCombinerHelperRuleConfig {
  SparseBitVector<> DisabledRules;

public:
  bool parseCommandLineOption();
  bool isRuleDisabled(unsigned ID) const;
  bool setRuleEnabled(StringRef RuleIdentifier);
  bool setRuleDisabled(StringRef RuleIdentifier);

};

class AMDGPUGenPreLegalizerCombinerHelper {
 const AMDGPUGenPreLegalizerCombinerHelperRuleConfig *RuleConfig;

public:
  template<typename ... Args>AMDGPUGenPreLegalizerCombinerHelper(const AMDGPUGenPreLegalizerCombinerHelperRuleConfig &RuleConfig, Args &&... args) : RuleConfig(&RuleConfig) {}

  bool tryCombineAll(
    GISelChangeObserver &Observer,
    MachineInstr &MI,
    MachineIRBuilder &B,
    CombinerHelper &Helper) const;
};

static Optional<uint64_t> getRuleIdxForIdentifier(StringRef RuleIdentifier) {
  uint64_t I;
  // getAtInteger(...) returns false on success
  bool Parsed = !RuleIdentifier.getAsInteger(0, I);
  if (Parsed)
    return I;

#ifndef NDEBUG
  switch (RuleIdentifier.size()) {
  default: break;
  case 9:	 // 2 strings to match.
    switch (RuleIdentifier[0]) {
    default: break;
    case 'c':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "opy_prop", 8) != 0)
        break;
      return 0;	 // "copy_prop"
    case 't':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "runc_shl", 8) != 0)
        break;
      return 43;	 // "trunc_shl"
    }
    break;
  case 10:	 // 3 strings to match.
    switch (RuleIdentifier[0]) {
    default: break;
    case 'i':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "2p_to_p2i", 9) != 0)
        break;
      return 21;	 // "i2p_to_p2i"
    case 'm':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "ul_to_shl", 9) != 0)
        break;
      return 1;	 // "mul_to_shl"
    case 'p':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "2i_to_i2p", 9) != 0)
        break;
      return 20;	 // "p2i_to_i2p"
    }
    break;
  case 11:	 // 1 string to match.
    if (memcmp(RuleIdentifier.data()+0, "unmerge_cst", 11) != 0)
      break;
    return 39;	 // "unmerge_cst"
  case 12:	 // 2 strings to match.
    switch (RuleIdentifier[0]) {
    default: break;
    case 'e':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "xt_ext_fold", 11) != 0)
        break;
      return 34;	 // "ext_ext_fold"
    case 'n':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "ot_cmp_fold", 11) != 0)
        break;
      return 35;	 // "not_cmp_fold"
    }
    break;
  case 13:	 // 1 string to match.
    if (memcmp(RuleIdentifier.data()+0, "unmerge_merge", 13) != 0)
      break;
    return 37;	 // "unmerge_merge"
  case 14:	 // 6 strings to match.
    switch (RuleIdentifier[0]) {
    default: break;
    case 'b':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "inop_same_val", 13) != 0)
        break;
      return 17;	 // "binop_same_val"
    case 'c':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "onstant_fp_op", 13) != 0)
        break;
      return 44;	 // "constant_fp_op"
    case 'f':	 // 2 strings to match.
      switch (RuleIdentifier[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(RuleIdentifier.data()+2, "bs_fabs_fold", 12) != 0)
          break;
        return 38;	 // "fabs_fabs_fold"
      case 'n':	 // 1 string to match.
        if (memcmp(RuleIdentifier.data()+2, "eg_fneg_fold", 12) != 0)
          break;
        return 23;	 // "fneg_fneg_fold"
      }
      break;
    case 'm':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "ul_by_neg_one", 13) != 0)
        break;
      return 3;	 // "mul_by_neg_one"
    case 't':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "runc_ext_fold", 13) != 0)
        break;
      return 42;	 // "trunc_ext_fold"
    }
    break;
  case 15:	 // 2 strings to match.
    switch (RuleIdentifier[0]) {
    default: break;
    case 'e':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "xtending_loads", 14) != 0)
        break;
      return 5;	 // "extending_loads"
    case 's':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "elect_same_val", 14) != 0)
        break;
      return 15;	 // "select_same_val"
    }
    break;
  case 16:	 // 3 strings to match.
    switch (RuleIdentifier[0]) {
    default: break;
    case 'a':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "nd_trivial_mask", 15) != 0)
        break;
      return 32;	 // "and_trivial_mask"
    case 's':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "elect_undef_cmp", 15) != 0)
        break;
      return 30;	 // "select_undef_cmp"
    case 'u':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "ndef_to_fp_zero", 15) != 0)
        break;
      return 7;	 // "undef_to_fp_zero"
    }
    break;
  case 17:	 // 5 strings to match.
    switch (RuleIdentifier[0]) {
    default: break;
    case 'a':	 // 2 strings to match.
      switch (RuleIdentifier[1]) {
      default: break;
      case 'd':	 // 1 string to match.
        if (memcmp(RuleIdentifier.data()+2, "d_p2i_to_ptradd", 15) != 0)
          break;
        return 2;	 // "add_p2i_to_ptradd"
      case 'n':	 // 1 string to match.
        if (memcmp(RuleIdentifier.data()+2, "yext_trunc_fold", 15) != 0)
          break;
        return 22;	 // "anyext_trunc_fold"
      }
      break;
    case 'e':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "rase_undef_store", 16) != 0)
        break;
      return 14;	 // "erase_undef_store"
    case 'p':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "tr_add_with_zero", 16) != 0)
        break;
      return 47;	 // "ptr_add_with_zero"
    case 'u':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "ndef_to_int_zero", 16) != 0)
        break;
      return 8;	 // "undef_to_int_zero"
    }
    break;
  case 18:	 // 3 strings to match.
    switch (RuleIdentifier[0]) {
    default: break;
    case 'b':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "inop_left_to_zero", 17) != 0)
        break;
      return 18;	 // "binop_left_to_zero"
    case 'r':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "ight_identity_one", 17) != 0)
        break;
      return 24;	 // "right_identity_one"
    case 's':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "ext_inreg_of_load", 17) != 0)
        break;
      return 28;	 // "sext_inreg_of_load"
    }
    break;
  case 19:	 // 6 strings to match.
    switch (RuleIdentifier[0]) {
    default: break;
    case 'b':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "inop_right_to_zero", 18) != 0)
        break;
      return 19;	 // "binop_right_to_zero"
    case 'c':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "onst_ptradd_to_i2p", 18) != 0)
        break;
      return 45;	 // "const_ptradd_to_i2p"
    case 'p':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "tr_add_immed_chain", 18) != 0)
        break;
      return 4;	 // "ptr_add_immed_chain"
    case 'r':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "ight_identity_zero", 18) != 0)
        break;
      return 16;	 // "right_identity_zero"
    case 's':	 // 2 strings to match.
      switch (RuleIdentifier[1]) {
      default: break;
      case 'e':	 // 1 string to match.
        if (memcmp(RuleIdentifier.data()+2, "lect_constant_cmp", 17) != 0)
          break;
        return 31;	 // "select_constant_cmp"
      case 'i':	 // 1 string to match.
        if (memcmp(RuleIdentifier.data()+2, "mplify_add_to_sub", 17) != 0)
          break;
        return 25;	 // "simplify_add_to_sub"
      }
      break;
    }
    break;
  case 20:	 // 3 strings to match.
    switch (RuleIdentifier[0]) {
    default: break;
    case 'r':	 // 2 strings to match.
      if (memcmp(RuleIdentifier.data()+1, "edu", 3) != 0)
        break;
      switch (RuleIdentifier[4]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(RuleIdentifier.data()+5, "e_shl_of_extend", 15) != 0)
          break;
        return 29;	 // "reduce_shl_of_extend"
      case 'n':	 // 1 string to match.
        if (memcmp(RuleIdentifier.data()+5, "dant_sext_inreg", 15) != 0)
          break;
        return 33;	 // "redundant_sext_inreg"
      }
      break;
    case 'u':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "nmerge_zext_to_zext", 19) != 0)
        break;
      return 41;	 // "unmerge_zext_to_zext"
    }
    break;
  case 21:	 // 2 strings to match.
    if (memcmp(RuleIdentifier.data()+0, "un", 2) != 0)
      break;
    switch (RuleIdentifier[2]) {
    default: break;
    case 'd':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+3, "ef_to_negative_one", 18) != 0)
        break;
      return 9;	 // "undef_to_negative_one"
    case 'm':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+3, "erge_dead_to_trunc", 18) != 0)
        break;
      return 40;	 // "unmerge_dead_to_trunc"
    }
    break;
  case 22:	 // 2 strings to match.
    switch (RuleIdentifier[0]) {
    default: break;
    case 'p':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "ropagate_undef_any_op", 21) != 0)
        break;
      return 11;	 // "propagate_undef_any_op"
    case 's':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "hl_ashr_to_sext_inreg", 21) != 0)
        break;
      return 27;	 // "shl_ashr_to_sext_inreg"
    }
    break;
  case 23:	 // 1 string to match.
    if (memcmp(RuleIdentifier.data()+0, "propagate_undef_all_ops", 23) != 0)
      break;
    return 12;	 // "propagate_undef_all_ops"
  case 24:	 // 2 strings to match.
    switch (RuleIdentifier[0]) {
    default: break;
    case 'b':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "inop_left_undef_to_zero", 23) != 0)
        break;
      return 10;	 // "binop_left_undef_to_zero"
    case 'x':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "or_of_and_with_same_reg", 23) != 0)
        break;
      return 46;	 // "xor_of_and_with_same_reg"
    }
    break;
  case 26:	 // 1 string to match.
    if (memcmp(RuleIdentifier.data()+0, "combine_indexed_load_store", 26) != 0)
      break;
    return 6;	 // "combine_indexed_load_store"
  case 28:	 // 2 strings to match.
    switch (RuleIdentifier[0]) {
    default: break;
    case 'o':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "pt_brcond_by_inverting_cond", 27) != 0)
        break;
      return 36;	 // "opt_brcond_by_inverting_cond"
    case 'p':	 // 1 string to match.
      if (memcmp(RuleIdentifier.data()+1, "ropagate_undef_shuffle_mask", 27) != 0)
        break;
      return 13;	 // "propagate_undef_shuffle_mask"
    }
    break;
  case 37:	 // 1 string to match.
    if (memcmp(RuleIdentifier.data()+0, "hoist_logic_op_with_same_opcode_hands", 37) != 0)
      break;
    return 26;	 // "hoist_logic_op_with_same_opcode_hands"
  }
#endif // ifndef NDEBUG

  return None;
}
static Optional<std::pair<uint64_t, uint64_t>> getRuleRangeForIdentifier(StringRef RuleIdentifier) {
  std::pair<StringRef, StringRef> RangePair = RuleIdentifier.split('-');
  if (!RangePair.second.empty()) {
    const auto First = getRuleIdxForIdentifier(RangePair.first);
    const auto Last = getRuleIdxForIdentifier(RangePair.second);
    if (!First.hasValue() || !Last.hasValue())
      return None;
    if (First >= Last)
      report_fatal_error("Beginning of range should be before end of range");
    return {{ *First, *Last + 1 }};
  } else if (RangePair.first == "*") {
    return {{ 0, 48 }};
  } else {
    const auto I = getRuleIdxForIdentifier(RangePair.first);
    if (!I.hasValue())
      return None;
    return {{*I, *I + 1}};
  }
  return None;
}

bool AMDGPUGenPreLegalizerCombinerHelperRuleConfig::setRuleEnabled(StringRef RuleIdentifier) {
  auto MaybeRange = getRuleRangeForIdentifier(RuleIdentifier);
  if(!MaybeRange.hasValue())
    return false;
  for (auto I = MaybeRange->first; I < MaybeRange->second; ++I)
    DisabledRules.reset(I);
  return true;
}

bool AMDGPUGenPreLegalizerCombinerHelperRuleConfig::setRuleDisabled(StringRef RuleIdentifier) {
  auto MaybeRange = getRuleRangeForIdentifier(RuleIdentifier);
  if(!MaybeRange.hasValue())
    return false;
  for (auto I = MaybeRange->first; I < MaybeRange->second; ++I)
    DisabledRules.set(I);
  return true;
}

bool AMDGPUGenPreLegalizerCombinerHelperRuleConfig::isRuleDisabled(unsigned RuleID) const {
  return DisabledRules.test(RuleID);
}
#endif // ifdef AMDGPUPRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H

#ifdef AMDGPUPRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP

std::vector<std::string> AMDGPUPreLegalizerCombinerHelperOption;
cl::list<std::string> AMDGPUPreLegalizerCombinerHelperDisableOption(
    "amdgpuprelegalizercombinerhelper-disable-rule",
    cl::desc("Disable one or more combiner rules temporarily in the AMDGPUPreLegalizerCombinerHelper pass"),
    cl::CommaSeparated,
    cl::Hidden,
    cl::cat(GICombinerOptionCategory),
    cl::callback([](const std::string &Str) {
      AMDGPUPreLegalizerCombinerHelperOption.push_back(Str);
    }));
cl::list<std::string> AMDGPUPreLegalizerCombinerHelperOnlyEnableOption(
    "amdgpuprelegalizercombinerhelper-only-enable-rule",
    cl::desc("Disable all rules in the AMDGPUPreLegalizerCombinerHelper pass then re-enable the specified ones"),
    cl::Hidden,
    cl::cat(GICombinerOptionCategory),
    cl::callback([](const std::string &CommaSeparatedArg) {
      StringRef Str = CommaSeparatedArg;
      AMDGPUPreLegalizerCombinerHelperOption.push_back("*");
      do {
        auto X = Str.split(",");
        AMDGPUPreLegalizerCombinerHelperOption.push_back(("!" + X.first).str());
        Str = X.second;
      } while (!Str.empty());
    }));

bool AMDGPUGenPreLegalizerCombinerHelperRuleConfig::parseCommandLineOption() {
  for (StringRef Identifier : AMDGPUPreLegalizerCombinerHelperOption) {
    bool Enabled = Identifier.consume_front("!");
    if (Enabled && !setRuleEnabled(Identifier))
      return false;
    if (!Enabled && !setRuleDisabled(Identifier))
      return false;
  }
  return true;
}

bool AMDGPUGenPreLegalizerCombinerHelper::tryCombineAll(
    GISelChangeObserver &Observer,
    MachineInstr &MI,
    MachineIRBuilder &B,
    CombinerHelper &Helper) const {
  MachineBasicBlock *MBB = MI.getParent();
  MachineFunction *MF = MBB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  SmallVector<MachineInstr *, 8> MIs = { &MI };

  (void)MBB; (void)MF; (void)MRI; (void)RuleConfig;

  // Match data
  unsigned MatchData1;
  std::pair<Register, bool> MatchData2;
  PtrAddChain MatchData4;
  PreferredTuple MatchData5;
  IndexedLoadStoreMatchInfo MatchData6;
  Register MatchData20;
  Register MatchData21;
  Register MatchData22;
  Register MatchData23;
  std::tuple<Register, Register> MatchData25;
  InstructionStepsMatchInfo MatchData26;
  std::tuple<Register, int64_t> MatchData27;
  std::tuple<Register, unsigned> MatchData28;
  RegisterImmPair MatchData29;
  unsigned MatchData31;
  Register MatchData32;
  std::tuple<Register, unsigned> MatchData34;
  SmallVector<Register, 4> MatchData35;
  SmallVector<Register, 8> MatchData37;
  Register MatchData38;
  SmallVector<APInt, 8> MatchData39;
  std::pair<Register, unsigned> MatchData42;
  std::pair<Register, Register> MatchData43;
  Optional<APFloat> MatchData44;
  int64_t MatchData45;
  std::pair<Register, Register> MatchData46;

  int Partition = -1;
  Partition = -1;
  switch (MIs[0]->getOpcode()) {
  case TargetOpcode::COPY: Partition = 0; break;
  case TargetOpcode::G_MUL: Partition = 1; break;
  case TargetOpcode::G_ADD: Partition = 2; break;
  case TargetOpcode::G_PTR_ADD: Partition = 3; break;
  case TargetOpcode::G_LOAD: Partition = 4; break;
  case TargetOpcode::G_SEXTLOAD: Partition = 5; break;
  case TargetOpcode::G_ZEXTLOAD: Partition = 6; break;
  case TargetOpcode::G_STORE: Partition = 7; break;
  case TargetOpcode::G_UITOFP: Partition = 8; break;
  case TargetOpcode::G_SITOFP: Partition = 9; break;
  case TargetOpcode::G_AND: Partition = 10; break;
  case TargetOpcode::G_OR: Partition = 11; break;
  case TargetOpcode::G_SHL: Partition = 12; break;
  case TargetOpcode::G_FPTOSI: Partition = 13; break;
  case TargetOpcode::G_FPTOUI: Partition = 14; break;
  case TargetOpcode::G_SUB: Partition = 15; break;
  case TargetOpcode::G_XOR: Partition = 16; break;
  case TargetOpcode::G_TRUNC: Partition = 17; break;
  case TargetOpcode::G_SHUFFLE_VECTOR: Partition = 18; break;
  case TargetOpcode::G_SELECT: Partition = 19; break;
  case TargetOpcode::G_ASHR: Partition = 20; break;
  case TargetOpcode::G_LSHR: Partition = 21; break;
  case TargetOpcode::G_SDIV: Partition = 22; break;
  case TargetOpcode::G_UDIV: Partition = 23; break;
  case TargetOpcode::G_SREM: Partition = 24; break;
  case TargetOpcode::G_UREM: Partition = 25; break;
  case TargetOpcode::G_INTTOPTR: Partition = 26; break;
  case TargetOpcode::G_PTRTOINT: Partition = 27; break;
  case TargetOpcode::G_ANYEXT: Partition = 28; break;
  case TargetOpcode::G_FNEG: Partition = 29; break;
  case TargetOpcode::G_SEXT_INREG: Partition = 30; break;
  case TargetOpcode::G_SEXT: Partition = 31; break;
  case TargetOpcode::G_ZEXT: Partition = 32; break;
  case TargetOpcode::G_BR: Partition = 33; break;
  case TargetOpcode::G_UNMERGE_VALUES: Partition = 34; break;
  case TargetOpcode::G_FABS: Partition = 35; break;
  case TargetOpcode::G_FPTRUNC: Partition = 36; break;
  case TargetOpcode::G_FSQRT: Partition = 37; break;
  case TargetOpcode::G_FLOG2: Partition = 38; break;
  }
  // Default case but without conflicting with potential default case in selection.
  if (Partition == -1) return false;
  if (Partition == 0 /* TargetOpcode::COPY */) {
    // Leaf name: copy_prop
    // Rule: copy_prop
    if (!RuleConfig->isRuleDisabled(0)) {
      if (1
          && [&]() {
           return Helper.matchCombineCopy(*MIs[0]); 
          return true;
      }()) {
        Helper.applyCombineCopy(*MIs[0]); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 1 /* TargetOpcode::G_MUL */) {
    // Leaf name: mul_to_shl
    // Rule: mul_to_shl
    if (!RuleConfig->isRuleDisabled(1)) {
      if (1
          && [&]() {
           return Helper.matchCombineMulToShl(*MIs[0], MatchData1); 
          return true;
      }()) {
        Helper.applyCombineMulToShl(*MIs[0], MatchData1); 
        return true;
      }
    }
    // Leaf name: mul_by_neg_one
    // Rule: mul_by_neg_one
    if (!RuleConfig->isRuleDisabled(3)) {
      if (1
          && [&]() {
           return Helper.matchConstantOp(MIs[0]->getOperand(2), -1); 
          return true;
      }()) {
        return Helper.applyCombineMulByNegativeOne(*MIs[0]); 
        return true;
      }
    }
    // Leaf name: undef_to_int_zero
    // Rule: undef_to_int_zero
    if (!RuleConfig->isRuleDisabled(8)) {
      if (1
          && [&]() {
           return Helper.matchAnyExplicitUseIsUndef(*MIs[0]); 
          return true;
      }()) {
        Helper.replaceInstWithConstant(*MIs[0], 0); 
        return true;
      }
    }
    // Leaf name: binop_right_to_zero
    // Rule: binop_right_to_zero
    if (!RuleConfig->isRuleDisabled(19)) {
      if (1
          && [&]() {
           return Helper.matchOperandIsZero(*MIs[0], 2); 
          return true;
      }()) {
        return Helper.replaceSingleDefInstWithOperand(*MIs[0], 2); 
        return true;
      }
    }
    // Leaf name: right_identity_one
    // Rule: right_identity_one
    if (!RuleConfig->isRuleDisabled(24)) {
      if (1
          && [&]() {
           return Helper.matchConstantOp(MIs[0]->getOperand(2), 1); 
          return true;
      }()) {
        return Helper.replaceSingleDefInstWithOperand(*MIs[0], 1); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 2 /* TargetOpcode::G_ADD */) {
    // Leaf name: add_p2i_to_ptradd
    // Rule: add_p2i_to_ptradd
    if (!RuleConfig->isRuleDisabled(2)) {
      if (1
          && [&]() {
           return Helper.matchCombineAddP2IToPtrAdd(*MIs[0], MatchData2); 
          return true;
      }()) {
        return Helper.applyCombineAddP2IToPtrAdd(*MIs[0], MatchData2); 
        return true;
      }
    }
    // Leaf name: propagate_undef_any_op
    // Rule: propagate_undef_any_op
    if (!RuleConfig->isRuleDisabled(11)) {
      if (1
          && [&]() {
           return Helper.matchAnyExplicitUseIsUndef(*MIs[0]); 
          return true;
      }()) {
        Helper.replaceInstWithUndef(*MIs[0]); 
        return true;
      }
    }
    // Leaf name: right_identity_zero
    // Rule: right_identity_zero
    if (!RuleConfig->isRuleDisabled(16)) {
      if (1
          && [&]() {
           return Helper.matchConstantOp(MIs[0]->getOperand(2), 0); 
          return true;
      }()) {
        return Helper.replaceSingleDefInstWithOperand(*MIs[0], 1); 
        return true;
      }
    }
    // Leaf name: simplify_add_to_sub
    // Rule: simplify_add_to_sub
    if (!RuleConfig->isRuleDisabled(25)) {
      if (1
          && [&]() {
           return Helper.matchSimplifyAddToSub(*MIs[0], MatchData25); 
          return true;
      }()) {
        return Helper.applySimplifyAddToSub(*MIs[0], MatchData25);
        return true;
      }
    }
    return false;
  }
  if (Partition == 3 /* TargetOpcode::G_PTR_ADD */) {
    // Leaf name: ptr_add_immed_chain
    // Rule: ptr_add_immed_chain
    if (!RuleConfig->isRuleDisabled(4)) {
      if (1
          && [&]() {
           return Helper.matchPtrAddImmedChain(*MIs[0], MatchData4); 
          return true;
      }()) {
        Helper.applyPtrAddImmedChain(*MIs[0], MatchData4); 
        return true;
      }
    }
    // Leaf name: const_ptradd_to_i2p
    // Rule: const_ptradd_to_i2p
    if (!RuleConfig->isRuleDisabled(45)) {
      if (1
          && [&]() {
           return Helper.matchCombineConstPtrAddToI2P(*MIs[0], MatchData45); 
          return true;
      }()) {
        return Helper.applyCombineConstPtrAddToI2P(*MIs[0], MatchData45); 
        return true;
      }
    }
    // Leaf name: ptr_add_with_zero
    // Rule: ptr_add_with_zero
    if (!RuleConfig->isRuleDisabled(47)) {
      if (1
          && [&]() {
           return Helper.matchPtrAddZero(*MIs[0]); 
          return true;
      }()) {
        return Helper.applyPtrAddZero(*MIs[0]); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 4 /* TargetOpcode::G_LOAD */) {
    // Leaf name: extending_loads
    // Rule: extending_loads
    if (!RuleConfig->isRuleDisabled(5)) {
      if (1
          && [&]() {
           return Helper.matchCombineExtendingLoads(*MIs[0], MatchData5); 
          return true;
      }()) {
        Helper.applyCombineExtendingLoads(*MIs[0], MatchData5); 
        return true;
      }
    }
    // Leaf name: combine_indexed_load_store
    // Rule: combine_indexed_load_store
    if (!RuleConfig->isRuleDisabled(6)) {
      if (1
          && [&]() {
           return Helper.matchCombineIndexedLoadStore(*MIs[0], MatchData6); 
          return true;
      }()) {
        Helper.applyCombineIndexedLoadStore(*MIs[0], MatchData6); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 5 /* TargetOpcode::G_SEXTLOAD */) {
    // Leaf name: extending_loads
    // Rule: extending_loads
    if (!RuleConfig->isRuleDisabled(5)) {
      if (1
          && [&]() {
           return Helper.matchCombineExtendingLoads(*MIs[0], MatchData5); 
          return true;
      }()) {
        Helper.applyCombineExtendingLoads(*MIs[0], MatchData5); 
        return true;
      }
    }
    // Leaf name: combine_indexed_load_store
    // Rule: combine_indexed_load_store
    if (!RuleConfig->isRuleDisabled(6)) {
      if (1
          && [&]() {
           return Helper.matchCombineIndexedLoadStore(*MIs[0], MatchData6); 
          return true;
      }()) {
        Helper.applyCombineIndexedLoadStore(*MIs[0], MatchData6); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 6 /* TargetOpcode::G_ZEXTLOAD */) {
    // Leaf name: extending_loads
    // Rule: extending_loads
    if (!RuleConfig->isRuleDisabled(5)) {
      if (1
          && [&]() {
           return Helper.matchCombineExtendingLoads(*MIs[0], MatchData5); 
          return true;
      }()) {
        Helper.applyCombineExtendingLoads(*MIs[0], MatchData5); 
        return true;
      }
    }
    // Leaf name: combine_indexed_load_store
    // Rule: combine_indexed_load_store
    if (!RuleConfig->isRuleDisabled(6)) {
      if (1
          && [&]() {
           return Helper.matchCombineIndexedLoadStore(*MIs[0], MatchData6); 
          return true;
      }()) {
        Helper.applyCombineIndexedLoadStore(*MIs[0], MatchData6); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 7 /* TargetOpcode::G_STORE */) {
    // Leaf name: combine_indexed_load_store
    // Rule: combine_indexed_load_store
    if (!RuleConfig->isRuleDisabled(6)) {
      if (1
          && [&]() {
           return Helper.matchCombineIndexedLoadStore(*MIs[0], MatchData6); 
          return true;
      }()) {
        Helper.applyCombineIndexedLoadStore(*MIs[0], MatchData6); 
        return true;
      }
    }
    // Leaf name: erase_undef_store
    // Rule: erase_undef_store
    if (!RuleConfig->isRuleDisabled(14)) {
      if (1
          && [&]() {
           return Helper.matchUndefStore(*MIs[0]); 
          return true;
      }()) {
        return Helper.eraseInst(*MIs[0]); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 8 /* TargetOpcode::G_UITOFP */) {
    // Leaf name: undef_to_fp_zero
    // Rule: undef_to_fp_zero
    if (!RuleConfig->isRuleDisabled(7)) {
      if (1
          && [&]() {
           return Helper.matchAnyExplicitUseIsUndef(*MIs[0]); 
          return true;
      }()) {
        Helper.replaceInstWithFConstant(*MIs[0], 0.0); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 9 /* TargetOpcode::G_SITOFP */) {
    // Leaf name: undef_to_fp_zero
    // Rule: undef_to_fp_zero
    if (!RuleConfig->isRuleDisabled(7)) {
      if (1
          && [&]() {
           return Helper.matchAnyExplicitUseIsUndef(*MIs[0]); 
          return true;
      }()) {
        Helper.replaceInstWithFConstant(*MIs[0], 0.0); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 10 /* TargetOpcode::G_AND */) {
    // Leaf name: undef_to_int_zero
    // Rule: undef_to_int_zero
    if (!RuleConfig->isRuleDisabled(8)) {
      if (1
          && [&]() {
           return Helper.matchAnyExplicitUseIsUndef(*MIs[0]); 
          return true;
      }()) {
        Helper.replaceInstWithConstant(*MIs[0], 0); 
        return true;
      }
    }
    // Leaf name: binop_same_val
    // Rule: binop_same_val
    if (!RuleConfig->isRuleDisabled(17)) {
      if (1
          && [&]() {
           return Helper.matchBinOpSameVal(*MIs[0]); 
          return true;
      }()) {
        return Helper.replaceSingleDefInstWithOperand(*MIs[0], 1); 
        return true;
      }
    }
    // Leaf name: hoist_logic_op_with_same_opcode_hands
    // Rule: hoist_logic_op_with_same_opcode_hands
    if (!RuleConfig->isRuleDisabled(26)) {
      if (1
          && [&]() {
           return Helper.matchHoistLogicOpWithSameOpcodeHands(*MIs[0], MatchData26); 
          return true;
      }()) {
        return Helper.applyBuildInstructionSteps(*MIs[0], MatchData26);
        return true;
      }
    }
    // Leaf name: and_trivial_mask
    // Rule: and_trivial_mask
    if (!RuleConfig->isRuleDisabled(32)) {
      if (1
          && [&]() {
           return Helper.matchAndWithTrivialMask(*MIs[0], MatchData32); 
          return true;
      }()) {
        return Helper.replaceSingleDefInstWithReg(*MIs[0], MatchData32); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 11 /* TargetOpcode::G_OR */) {
    // Leaf name: undef_to_negative_one
    // Rule: undef_to_negative_one
    if (!RuleConfig->isRuleDisabled(9)) {
      if (1
          && [&]() {
           return Helper.matchAnyExplicitUseIsUndef(*MIs[0]); 
          return true;
      }()) {
        Helper.replaceInstWithConstant(*MIs[0], -1); 
        return true;
      }
    }
    // Leaf name: right_identity_zero
    // Rule: right_identity_zero
    if (!RuleConfig->isRuleDisabled(16)) {
      if (1
          && [&]() {
           return Helper.matchConstantOp(MIs[0]->getOperand(2), 0); 
          return true;
      }()) {
        return Helper.replaceSingleDefInstWithOperand(*MIs[0], 1); 
        return true;
      }
    }
    // Leaf name: binop_same_val
    // Rule: binop_same_val
    if (!RuleConfig->isRuleDisabled(17)) {
      if (1
          && [&]() {
           return Helper.matchBinOpSameVal(*MIs[0]); 
          return true;
      }()) {
        return Helper.replaceSingleDefInstWithOperand(*MIs[0], 1); 
        return true;
      }
    }
    // Leaf name: hoist_logic_op_with_same_opcode_hands
    // Rule: hoist_logic_op_with_same_opcode_hands
    if (!RuleConfig->isRuleDisabled(26)) {
      if (1
          && [&]() {
           return Helper.matchHoistLogicOpWithSameOpcodeHands(*MIs[0], MatchData26); 
          return true;
      }()) {
        return Helper.applyBuildInstructionSteps(*MIs[0], MatchData26);
        return true;
      }
    }
    return false;
  }
  if (Partition == 12 /* TargetOpcode::G_SHL */) {
    // Leaf name: binop_left_undef_to_zero
    // Rule: binop_left_undef_to_zero
    if (!RuleConfig->isRuleDisabled(10)) {
      if (1
          && [&]() {
           return Helper.matchOperandIsUndef(*MIs[0], 1); 
          return true;
      }()) {
        Helper.replaceInstWithConstant(*MIs[0], 0); 
        return true;
      }
    }
    // Leaf name: right_identity_zero
    // Rule: right_identity_zero
    if (!RuleConfig->isRuleDisabled(16)) {
      if (1
          && [&]() {
           return Helper.matchConstantOp(MIs[0]->getOperand(2), 0); 
          return true;
      }()) {
        return Helper.replaceSingleDefInstWithOperand(*MIs[0], 1); 
        return true;
      }
    }
    // Leaf name: reduce_shl_of_extend
    // Rule: reduce_shl_of_extend
    if (!RuleConfig->isRuleDisabled(29)) {
      if (1
          && [&]() {
           return Helper.matchCombineShlOfExtend(*MIs[0], MatchData29); 
          return true;
      }()) {
        Helper.applyCombineShlOfExtend(*MIs[0], MatchData29); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 13 /* TargetOpcode::G_FPTOSI */) {
    // Leaf name: propagate_undef_any_op
    // Rule: propagate_undef_any_op
    if (!RuleConfig->isRuleDisabled(11)) {
      if (1
          && [&]() {
           return Helper.matchAnyExplicitUseIsUndef(*MIs[0]); 
          return true;
      }()) {
        Helper.replaceInstWithUndef(*MIs[0]); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 14 /* TargetOpcode::G_FPTOUI */) {
    // Leaf name: propagate_undef_any_op
    // Rule: propagate_undef_any_op
    if (!RuleConfig->isRuleDisabled(11)) {
      if (1
          && [&]() {
           return Helper.matchAnyExplicitUseIsUndef(*MIs[0]); 
          return true;
      }()) {
        Helper.replaceInstWithUndef(*MIs[0]); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 15 /* TargetOpcode::G_SUB */) {
    // Leaf name: propagate_undef_any_op
    // Rule: propagate_undef_any_op
    if (!RuleConfig->isRuleDisabled(11)) {
      if (1
          && [&]() {
           return Helper.matchAnyExplicitUseIsUndef(*MIs[0]); 
          return true;
      }()) {
        Helper.replaceInstWithUndef(*MIs[0]); 
        return true;
      }
    }
    // Leaf name: right_identity_zero
    // Rule: right_identity_zero
    if (!RuleConfig->isRuleDisabled(16)) {
      if (1
          && [&]() {
           return Helper.matchConstantOp(MIs[0]->getOperand(2), 0); 
          return true;
      }()) {
        return Helper.replaceSingleDefInstWithOperand(*MIs[0], 1); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 16 /* TargetOpcode::G_XOR */) {
    // Leaf name: propagate_undef_any_op
    // Rule: propagate_undef_any_op
    if (!RuleConfig->isRuleDisabled(11)) {
      if (1
          && [&]() {
           return Helper.matchAnyExplicitUseIsUndef(*MIs[0]); 
          return true;
      }()) {
        Helper.replaceInstWithUndef(*MIs[0]); 
        return true;
      }
    }
    // Leaf name: right_identity_zero
    // Rule: right_identity_zero
    if (!RuleConfig->isRuleDisabled(16)) {
      if (1
          && [&]() {
           return Helper.matchConstantOp(MIs[0]->getOperand(2), 0); 
          return true;
      }()) {
        return Helper.replaceSingleDefInstWithOperand(*MIs[0], 1); 
        return true;
      }
    }
    // Leaf name: hoist_logic_op_with_same_opcode_hands
    // Rule: hoist_logic_op_with_same_opcode_hands
    if (!RuleConfig->isRuleDisabled(26)) {
      if (1
          && [&]() {
           return Helper.matchHoistLogicOpWithSameOpcodeHands(*MIs[0], MatchData26); 
          return true;
      }()) {
        return Helper.applyBuildInstructionSteps(*MIs[0], MatchData26);
        return true;
      }
    }
    // Leaf name: not_cmp_fold
    // Rule: not_cmp_fold
    if (!RuleConfig->isRuleDisabled(35)) {
      if (1
          && [&]() {
           return Helper.matchNotCmp(*MIs[0], MatchData35); 
          return true;
      }()) {
        return Helper.applyNotCmp(*MIs[0], MatchData35); 
        return true;
      }
    }
    // Leaf name: xor_of_and_with_same_reg
    // Rule: xor_of_and_with_same_reg
    if (!RuleConfig->isRuleDisabled(46)) {
      if (1
          && [&]() {
           return Helper.matchXorOfAndWithSameReg(*MIs[0], MatchData46); 
          return true;
      }()) {
        return Helper.applyXorOfAndWithSameReg(*MIs[0], MatchData46); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 17 /* TargetOpcode::G_TRUNC */) {
    // Leaf name: propagate_undef_any_op
    // Rule: propagate_undef_any_op
    if (!RuleConfig->isRuleDisabled(11)) {
      if (1
          && [&]() {
           return Helper.matchAnyExplicitUseIsUndef(*MIs[0]); 
          return true;
      }()) {
        Helper.replaceInstWithUndef(*MIs[0]); 
        return true;
      }
    }
    // Leaf name: trunc_ext_fold
    // Rule: trunc_ext_fold
    if (!RuleConfig->isRuleDisabled(42)) {
      if (1
          && [&]() {
           return Helper.matchCombineTruncOfExt(*MIs[0], MatchData42); 
          return true;
      }()) {
        return Helper.applyCombineTruncOfExt(*MIs[0], MatchData42); 
        return true;
      }
    }
    // Leaf name: trunc_shl
    // Rule: trunc_shl
    if (!RuleConfig->isRuleDisabled(43)) {
      if (1
          && [&]() {
           return Helper.matchCombineTruncOfShl(*MIs[0], MatchData43); 
          return true;
      }()) {
        return Helper.applyCombineTruncOfShl(*MIs[0], MatchData43); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 18 /* TargetOpcode::G_SHUFFLE_VECTOR */) {
    // Leaf name: propagate_undef_all_ops
    // Rule: propagate_undef_all_ops
    if (!RuleConfig->isRuleDisabled(12)) {
      if (1
          && [&]() {
           return Helper.matchAllExplicitUsesAreUndef(*MIs[0]); 
          return true;
      }()) {
        Helper.replaceInstWithUndef(*MIs[0]); 
        return true;
      }
    }
    // Leaf name: propagate_undef_shuffle_mask
    // Rule: propagate_undef_shuffle_mask
    if (!RuleConfig->isRuleDisabled(13)) {
      if (1
          && [&]() {
           return Helper.matchUndefShuffleVectorMask(*MIs[0]); 
          return true;
      }()) {
        Helper.replaceInstWithUndef(*MIs[0]); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 19 /* TargetOpcode::G_SELECT */) {
    // Leaf name: select_same_val
    // Rule: select_same_val
    if (!RuleConfig->isRuleDisabled(15)) {
      if (1
          && [&]() {
           return Helper.matchSelectSameVal(*MIs[0]); 
          return true;
      }()) {
        return Helper.replaceSingleDefInstWithOperand(*MIs[0], 2); 
        return true;
      }
    }
    // Leaf name: select_undef_cmp
    // Rule: select_undef_cmp
    if (!RuleConfig->isRuleDisabled(30)) {
      if (1
          && [&]() {
           return Helper.matchUndefSelectCmp(*MIs[0]); 
          return true;
      }()) {
        return Helper.replaceSingleDefInstWithOperand(*MIs[0], 2); 
        return true;
      }
    }
    // Leaf name: select_constant_cmp
    // Rule: select_constant_cmp
    if (!RuleConfig->isRuleDisabled(31)) {
      if (1
          && [&]() {
           return Helper.matchConstantSelectCmp(*MIs[0], MatchData31); 
          return true;
      }()) {
        return Helper.replaceSingleDefInstWithOperand(*MIs[0], MatchData31); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 20 /* TargetOpcode::G_ASHR */) {
    // Leaf name: right_identity_zero
    // Rule: right_identity_zero
    if (!RuleConfig->isRuleDisabled(16)) {
      if (1
          && [&]() {
           return Helper.matchConstantOp(MIs[0]->getOperand(2), 0); 
          return true;
      }()) {
        return Helper.replaceSingleDefInstWithOperand(*MIs[0], 1); 
        return true;
      }
    }
    // Leaf name: shl_ashr_to_sext_inreg
    // Rule: shl_ashr_to_sext_inreg
    if (!RuleConfig->isRuleDisabled(27)) {
      if (1
          && [&]() {
           return Helper.matchAshrShlToSextInreg(*MIs[0], MatchData27); 
          return true;
      }()) {
        return Helper.applyAshShlToSextInreg(*MIs[0], MatchData27);
        return true;
      }
    }
    return false;
  }
  if (Partition == 21 /* TargetOpcode::G_LSHR */) {
    // Leaf name: right_identity_zero
    // Rule: right_identity_zero
    if (!RuleConfig->isRuleDisabled(16)) {
      if (1
          && [&]() {
           return Helper.matchConstantOp(MIs[0]->getOperand(2), 0); 
          return true;
      }()) {
        return Helper.replaceSingleDefInstWithOperand(*MIs[0], 1); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 22 /* TargetOpcode::G_SDIV */) {
    // Leaf name: binop_left_to_zero
    // Rule: binop_left_to_zero
    if (!RuleConfig->isRuleDisabled(18)) {
      if (1
          && [&]() {
           return Helper.matchOperandIsZero(*MIs[0], 1); 
          return true;
      }()) {
        return Helper.replaceSingleDefInstWithOperand(*MIs[0], 1); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 23 /* TargetOpcode::G_UDIV */) {
    // Leaf name: binop_left_to_zero
    // Rule: binop_left_to_zero
    if (!RuleConfig->isRuleDisabled(18)) {
      if (1
          && [&]() {
           return Helper.matchOperandIsZero(*MIs[0], 1); 
          return true;
      }()) {
        return Helper.replaceSingleDefInstWithOperand(*MIs[0], 1); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 24 /* TargetOpcode::G_SREM */) {
    // Leaf name: binop_left_to_zero
    // Rule: binop_left_to_zero
    if (!RuleConfig->isRuleDisabled(18)) {
      if (1
          && [&]() {
           return Helper.matchOperandIsZero(*MIs[0], 1); 
          return true;
      }()) {
        return Helper.replaceSingleDefInstWithOperand(*MIs[0], 1); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 25 /* TargetOpcode::G_UREM */) {
    // Leaf name: binop_left_to_zero
    // Rule: binop_left_to_zero
    if (!RuleConfig->isRuleDisabled(18)) {
      if (1
          && [&]() {
           return Helper.matchOperandIsZero(*MIs[0], 1); 
          return true;
      }()) {
        return Helper.replaceSingleDefInstWithOperand(*MIs[0], 1); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 26 /* TargetOpcode::G_INTTOPTR */) {
    // Leaf name: p2i_to_i2p
    // Rule: p2i_to_i2p
    if (!RuleConfig->isRuleDisabled(20)) {
      if (1
          && [&]() {
           return Helper.matchCombineI2PToP2I(*MIs[0], MatchData20); 
          return true;
      }()) {
        return Helper.applyCombineI2PToP2I(*MIs[0], MatchData20); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 27 /* TargetOpcode::G_PTRTOINT */) {
    // Leaf name: i2p_to_p2i
    // Rule: i2p_to_p2i
    if (!RuleConfig->isRuleDisabled(21)) {
      if (1
          && [&]() {
           return Helper.matchCombineP2IToI2P(*MIs[0], MatchData21); 
          return true;
      }()) {
        return Helper.applyCombineP2IToI2P(*MIs[0], MatchData21); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 28 /* TargetOpcode::G_ANYEXT */) {
    // Leaf name: anyext_trunc_fold
    // Rule: anyext_trunc_fold
    if (!RuleConfig->isRuleDisabled(22)) {
      if (1
          && [&]() {
           return Helper.matchCombineAnyExtTrunc(*MIs[0], MatchData22); 
          return true;
      }()) {
        return Helper.applyCombineAnyExtTrunc(*MIs[0], MatchData22); 
        return true;
      }
    }
    // Leaf name: ext_ext_fold
    // Rule: ext_ext_fold
    if (!RuleConfig->isRuleDisabled(34)) {
      if (1
          && [&]() {
           return Helper.matchCombineExtOfExt(*MIs[0], MatchData34); 
          return true;
      }()) {
        return Helper.applyCombineExtOfExt(*MIs[0], MatchData34); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 29 /* TargetOpcode::G_FNEG */) {
    // Leaf name: fneg_fneg_fold
    // Rule: fneg_fneg_fold
    if (!RuleConfig->isRuleDisabled(23)) {
      if (1
          && [&]() {
           return Helper.matchCombineFNegOfFNeg(*MIs[0], MatchData23); 
          return true;
      }()) {
        return Helper.replaceSingleDefInstWithReg(*MIs[0], MatchData23); 
        return true;
      }
    }
    // Leaf name: constant_fp_op
    // Rule: constant_fp_op
    if (!RuleConfig->isRuleDisabled(44)) {
      if (1
          && [&]() {
           return Helper.matchCombineConstantFoldFpUnary(*MIs[0], MatchData44); 
          return true;
      }()) {
        return Helper.applyCombineConstantFoldFpUnary(*MIs[0], MatchData44); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 30 /* TargetOpcode::G_SEXT_INREG */) {
    // Leaf name: sext_inreg_of_load
    // Rule: sext_inreg_of_load
    if (!RuleConfig->isRuleDisabled(28)) {
      if (1
          && [&]() {
           return Helper.matchSextInRegOfLoad(*MIs[0], MatchData28); 
          return true;
      }()) {
        return Helper.applySextInRegOfLoad(*MIs[0], MatchData28); 
        return true;
      }
    }
    // Leaf name: redundant_sext_inreg
    // Rule: redundant_sext_inreg
    if (!RuleConfig->isRuleDisabled(33)) {
      if (1
          && [&]() {
           return Helper.matchRedundantSExtInReg(*MIs[0]); 
          return true;
      }()) {
        return Helper.replaceSingleDefInstWithOperand(*MIs[0], 1); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 31 /* TargetOpcode::G_SEXT */) {
    // Leaf name: ext_ext_fold
    // Rule: ext_ext_fold
    if (!RuleConfig->isRuleDisabled(34)) {
      if (1
          && [&]() {
           return Helper.matchCombineExtOfExt(*MIs[0], MatchData34); 
          return true;
      }()) {
        return Helper.applyCombineExtOfExt(*MIs[0], MatchData34); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 32 /* TargetOpcode::G_ZEXT */) {
    // Leaf name: ext_ext_fold
    // Rule: ext_ext_fold
    if (!RuleConfig->isRuleDisabled(34)) {
      if (1
          && [&]() {
           return Helper.matchCombineExtOfExt(*MIs[0], MatchData34); 
          return true;
      }()) {
        return Helper.applyCombineExtOfExt(*MIs[0], MatchData34); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 33 /* TargetOpcode::G_BR */) {
    // Leaf name: opt_brcond_by_inverting_cond
    // Rule: opt_brcond_by_inverting_cond
    if (!RuleConfig->isRuleDisabled(36)) {
      if (1
          && [&]() {
           return Helper.matchOptBrCondByInvertingCond(*MIs[0]); 
          return true;
      }()) {
        Helper.applyOptBrCondByInvertingCond(*MIs[0]); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 34 /* TargetOpcode::G_UNMERGE_VALUES */) {
    // Leaf name: unmerge_merge
    // Rule: unmerge_merge
    if (!RuleConfig->isRuleDisabled(37)) {
      if (1
          && [&]() {
           return Helper.matchCombineUnmergeMergeToPlainValues(*MIs[0], MatchData37); 
          return true;
      }()) {
        return Helper.applyCombineUnmergeMergeToPlainValues(*MIs[0], MatchData37); 
        return true;
      }
    }
    // Leaf name: unmerge_cst
    // Rule: unmerge_cst
    if (!RuleConfig->isRuleDisabled(39)) {
      if (1
          && [&]() {
           return Helper.matchCombineUnmergeConstant(*MIs[0], MatchData39); 
          return true;
      }()) {
        return Helper.applyCombineUnmergeConstant(*MIs[0], MatchData39); 
        return true;
      }
    }
    // Leaf name: unmerge_dead_to_trunc
    // Rule: unmerge_dead_to_trunc
    if (!RuleConfig->isRuleDisabled(40)) {
      if (1
          && [&]() {
           return Helper.matchCombineUnmergeWithDeadLanesToTrunc(*MIs[0]); 
          return true;
      }()) {
        return Helper.applyCombineUnmergeWithDeadLanesToTrunc(*MIs[0]); 
        return true;
      }
    }
    // Leaf name: unmerge_zext_to_zext
    // Rule: unmerge_zext_to_zext
    if (!RuleConfig->isRuleDisabled(41)) {
      if (1
          && [&]() {
           return Helper.matchCombineUnmergeZExtToZExt(*MIs[0]); 
          return true;
      }()) {
        return Helper.applyCombineUnmergeZExtToZExt(*MIs[0]); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 35 /* TargetOpcode::G_FABS */) {
    // Leaf name: fabs_fabs_fold
    // Rule: fabs_fabs_fold
    if (!RuleConfig->isRuleDisabled(38)) {
      if (1
          && [&]() {
           return Helper.matchCombineFAbsOfFAbs(*MIs[0], MatchData38); 
          return true;
      }()) {
        return Helper.applyCombineFAbsOfFAbs(*MIs[0], MatchData38); 
        return true;
      }
    }
    // Leaf name: constant_fp_op
    // Rule: constant_fp_op
    if (!RuleConfig->isRuleDisabled(44)) {
      if (1
          && [&]() {
           return Helper.matchCombineConstantFoldFpUnary(*MIs[0], MatchData44); 
          return true;
      }()) {
        return Helper.applyCombineConstantFoldFpUnary(*MIs[0], MatchData44); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 36 /* TargetOpcode::G_FPTRUNC */) {
    // Leaf name: constant_fp_op
    // Rule: constant_fp_op
    if (!RuleConfig->isRuleDisabled(44)) {
      if (1
          && [&]() {
           return Helper.matchCombineConstantFoldFpUnary(*MIs[0], MatchData44); 
          return true;
      }()) {
        return Helper.applyCombineConstantFoldFpUnary(*MIs[0], MatchData44); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 37 /* TargetOpcode::G_FSQRT */) {
    // Leaf name: constant_fp_op
    // Rule: constant_fp_op
    if (!RuleConfig->isRuleDisabled(44)) {
      if (1
          && [&]() {
           return Helper.matchCombineConstantFoldFpUnary(*MIs[0], MatchData44); 
          return true;
      }()) {
        return Helper.applyCombineConstantFoldFpUnary(*MIs[0], MatchData44); 
        return true;
      }
    }
    return false;
  }
  if (Partition == 38 /* TargetOpcode::G_FLOG2 */) {
    // Leaf name: constant_fp_op
    // Rule: constant_fp_op
    if (!RuleConfig->isRuleDisabled(44)) {
      if (1
          && [&]() {
           return Helper.matchCombineConstantFoldFpUnary(*MIs[0], MatchData44); 
          return true;
      }()) {
        return Helper.applyCombineConstantFoldFpUnary(*MIs[0], MatchData44); 
        return true;
      }
    }
    return false;
  }

  return false;
}
#endif // ifdef AMDGPUPRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP
