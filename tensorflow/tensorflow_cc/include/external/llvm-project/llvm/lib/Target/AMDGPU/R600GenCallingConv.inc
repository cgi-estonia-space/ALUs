/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Calling Convention Implementation Fragment                                 *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

static bool CC_R600(unsigned ValNo, MVT ValVT,
                    MVT LocVT, CCValAssign::LocInfo LocInfo,
                    ISD::ArgFlagsTy ArgFlags, CCState &State);


static bool CC_R600(unsigned ValNo, MVT ValVT,
                    MVT LocVT, CCValAssign::LocInfo LocInfo,
                    ISD::ArgFlagsTy ArgFlags, CCState &State) {

  if (ArgFlags.isInReg()) {
    if (LocVT == MVT::v4f32 ||
        LocVT == MVT::v4i32) {
      static const MCPhysReg RegList1[] = {
        R600::T0_XYZW, R600::T1_XYZW, R600::T2_XYZW, R600::T3_XYZW, R600::T4_XYZW, R600::T5_XYZW, R600::T6_XYZW, R600::T7_XYZW, R600::T8_XYZW, R600::T9_XYZW, R600::T10_XYZW, R600::T11_XYZW, R600::T12_XYZW, R600::T13_XYZW, R600::T14_XYZW, R600::T15_XYZW, R600::T16_XYZW, R600::T17_XYZW, R600::T18_XYZW, R600::T19_XYZW, R600::T20_XYZW, R600::T21_XYZW, R600::T22_XYZW, R600::T23_XYZW, R600::T24_XYZW, R600::T25_XYZW, R600::T26_XYZW, R600::T27_XYZW, R600::T28_XYZW, R600::T29_XYZW, R600::T30_XYZW, R600::T31_XYZW, R600::T32_XYZW
      };
      if (unsigned Reg = State.AllocateReg(RegList1)) {
        State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
        return false;
      }
    }
  }

  return true;  // CC didn't match.
}
