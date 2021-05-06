/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* DAG Instruction Selector for the R600 target                               *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

// *** NOTE: This file is #included into the middle of the target
// *** instruction selector class.  These functions are really methods.

// If GET_DAGISEL_DECL is #defined with any value, only function
// declarations will be included when this file is included.
// If GET_DAGISEL_BODY is #defined, its value should be the name of
// the instruction selector class. Function bodies will be emitted
// and each function's name will be qualified with the name of the
// class.
//
// When neither of the GET_DAGISEL* macros is defined, the functions
// are emitted inline.

#if defined(GET_DAGISEL_DECL) && defined(GET_DAGISEL_BODY)
#error GET_DAGISEL_DECL and GET_DAGISEL_BODY cannot be both defined, undef both for inline definitions
#endif

#ifdef GET_DAGISEL_BODY
#define LOCAL_DAGISEL_STRINGIZE(X) LOCAL_DAGISEL_STRINGIZE_(X)
#define LOCAL_DAGISEL_STRINGIZE_(X) #X
static_assert(sizeof(LOCAL_DAGISEL_STRINGIZE(GET_DAGISEL_BODY)) > 1,
   "GET_DAGISEL_BODY is empty: it should be defined with the class name");
#undef LOCAL_DAGISEL_STRINGIZE_
#undef LOCAL_DAGISEL_STRINGIZE
#endif

#if !defined(GET_DAGISEL_DECL) && !defined(GET_DAGISEL_BODY)
#define DAGISEL_INLINE 1
#else
#define DAGISEL_INLINE 0
#endif

#if !DAGISEL_INLINE
#define DAGISEL_CLASS_COLONCOLON GET_DAGISEL_BODY ::
#else
#define DAGISEL_CLASS_COLONCOLON
#endif

#ifdef GET_DAGISEL_DECL
void SelectCode(SDNode *N);
#endif
#if defined(GET_DAGISEL_BODY) || DAGISEL_INLINE
void DAGISEL_CLASS_COLONCOLON SelectCode(SDNode *N)
{
  // Some target values are emitted as 2 bytes, TARGET_VAL handles
  // this.
  #define TARGET_VAL(X) X & 255, unsigned(X) >> 8
  static const unsigned char MatcherTable[] = {
/*     0*/ OPC_SwitchOpcode /*103 cases */, 99|128,2/*355*/, TARGET_VAL(ISD::SRL),// ->360
/*     5*/  OPC_Scope, 11|128,1/*139*/, /*->147*/ // 2 children in Scope
/*     8*/   OPC_MoveChild0,
/*     9*/   OPC_CheckOpcode, TARGET_VAL(ISD::SHL),
/*    12*/   OPC_RecordChild0, // #0 = $src
/*    13*/   OPC_MoveChild1,
/*    14*/   OPC_CheckOpcode, TARGET_VAL(ISD::SUB),
/*    17*/   OPC_CheckChild0Integer, 32, 
/*    19*/   OPC_RecordChild1, // #1 = $width
/*    20*/   OPC_CheckType, MVT::i32,
/*    22*/   OPC_MoveParent,
/*    23*/   OPC_CheckPredicate, 0, // Predicate_shl_oneuse
/*    25*/   OPC_MoveParent,
/*    26*/   OPC_MoveChild1,
/*    27*/   OPC_CheckOpcode, TARGET_VAL(ISD::SUB),
/*    30*/   OPC_CheckChild0Integer, 32, 
/*    32*/   OPC_CheckChild1Same, 1,
/*    34*/   OPC_CheckType, MVT::i32,
/*    36*/   OPC_MoveParent,
/*    37*/   OPC_CheckType, MVT::i32,
/*    39*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*    41*/   OPC_EmitInteger, MVT::i32, 0, 
/*    44*/   OPC_EmitInteger, MVT::i32, 0, 
/*    47*/   OPC_EmitInteger, MVT::i32, 0, 
/*    50*/   OPC_EmitInteger, MVT::i32, 0, 
/*    53*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*    65*/   OPC_EmitInteger, MVT::i32, 0, 
/*    68*/   OPC_EmitNode1, TARGET_VAL(R600::MOV_IMM_I32), 0,
                 MVT::i32, 1/*#Ops*/, 7,  // Results = #8
/*    75*/   OPC_EmitInteger, MVT::i32, 0, 
/*    78*/   OPC_EmitInteger, MVT::i32, 0, 
/*    81*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*    93*/   OPC_EmitInteger, MVT::i32, 0, 
/*    96*/   OPC_EmitInteger, MVT::i32, 0, 
/*    99*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*   111*/   OPC_EmitInteger, MVT::i32, 1, 
/*   114*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*   117*/   OPC_EmitInteger, MVT::i32, 0, 
/*   120*/   OPC_EmitInteger, MVT::i32, 0, 
/*   123*/   OPC_MorphNodeTo1, TARGET_VAL(R600::BFE_UINT_eg), 0,
                 MVT::i32, 18/*#Ops*/, 2, 3, 0, 4, 5, 6, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 
             // Src: (srl:{ *:[i32] } (shl:{ *:[i32] } i32:{ *:[i32] }:$src, (sub:{ *:[i32] } 32:{ *:[i32] }, i32:{ *:[i32] }:$width))<<P:Predicate_shl_oneuse>>, (sub:{ *:[i32] } 32:{ *:[i32] }, i32:{ *:[i32] }:$width)) - Complexity = 23
             // Dst: (BFE_UINT_eg:{ *:[i32] } ?:{ *:[i32] }:$src, (MOV_IMM_I32:{ *:[i32] } 0:{ *:[i32] }), ?:{ *:[i32] }:$width)
/*   147*/  /*Scope*/ 82|128,1/*210*/, /*->359*/
/*   149*/   OPC_RecordChild0, // #0 = $src0
/*   150*/   OPC_RecordChild1, // #1 = $src1
/*   151*/   OPC_CheckChild1Type, MVT::i32,
/*   153*/   OPC_CheckType, MVT::i32,
/*   155*/   OPC_Scope, 100, /*->257*/ // 2 children in Scope
/*   157*/    OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/*   159*/    OPC_EmitInteger, MVT::i32, 0, 
/*   162*/    OPC_EmitInteger, MVT::i32, 0, 
/*   165*/    OPC_EmitInteger, MVT::i32, 1, 
/*   168*/    OPC_EmitInteger, MVT::i32, 0, 
/*   171*/    OPC_EmitInteger, MVT::i32, 0, 
/*   174*/    OPC_EmitInteger, MVT::i32, 0, 
/*   177*/    OPC_EmitInteger, MVT::i32, 0, 
/*   180*/    OPC_EmitInteger, MVT::i32, 0, 
/*   183*/    OPC_EmitInteger, MVT::i32, 0, 
/*   186*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*   198*/    OPC_EmitInteger, MVT::i32, 0, 
/*   201*/    OPC_EmitInteger, MVT::i32, 0, 
/*   204*/    OPC_EmitInteger, MVT::i32, 0, 
/*   207*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*   219*/    OPC_EmitInteger, MVT::i32, 1, 
/*   222*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*   225*/    OPC_EmitInteger, MVT::i32, 0, 
/*   228*/    OPC_EmitInteger, MVT::i32, 0, 
/*   231*/    OPC_MorphNodeTo1, TARGET_VAL(R600::LSHR_r600), 0,
                  MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
              // Src: (srl:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
              // Dst: (LSHR_r600:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/*   257*/   /*Scope*/ 100, /*->358*/
/*   258*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*   260*/    OPC_EmitInteger, MVT::i32, 0, 
/*   263*/    OPC_EmitInteger, MVT::i32, 0, 
/*   266*/    OPC_EmitInteger, MVT::i32, 1, 
/*   269*/    OPC_EmitInteger, MVT::i32, 0, 
/*   272*/    OPC_EmitInteger, MVT::i32, 0, 
/*   275*/    OPC_EmitInteger, MVT::i32, 0, 
/*   278*/    OPC_EmitInteger, MVT::i32, 0, 
/*   281*/    OPC_EmitInteger, MVT::i32, 0, 
/*   284*/    OPC_EmitInteger, MVT::i32, 0, 
/*   287*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*   299*/    OPC_EmitInteger, MVT::i32, 0, 
/*   302*/    OPC_EmitInteger, MVT::i32, 0, 
/*   305*/    OPC_EmitInteger, MVT::i32, 0, 
/*   308*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*   320*/    OPC_EmitInteger, MVT::i32, 1, 
/*   323*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*   326*/    OPC_EmitInteger, MVT::i32, 0, 
/*   329*/    OPC_EmitInteger, MVT::i32, 0, 
/*   332*/    OPC_MorphNodeTo1, TARGET_VAL(R600::LSHR_eg), 0,
                  MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
              // Src: (srl:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
              // Dst: (LSHR_eg:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/*   358*/   0, /*End of Scope*/
/*   359*/  0, /*End of Scope*/
/*   360*/ /*SwitchOpcode*/ 99|128,2/*355*/, TARGET_VAL(ISD::SRA),// ->719
/*   364*/  OPC_Scope, 11|128,1/*139*/, /*->506*/ // 2 children in Scope
/*   367*/   OPC_MoveChild0,
/*   368*/   OPC_CheckOpcode, TARGET_VAL(ISD::SHL),
/*   371*/   OPC_RecordChild0, // #0 = $src
/*   372*/   OPC_MoveChild1,
/*   373*/   OPC_CheckOpcode, TARGET_VAL(ISD::SUB),
/*   376*/   OPC_CheckChild0Integer, 32, 
/*   378*/   OPC_RecordChild1, // #1 = $width
/*   379*/   OPC_CheckType, MVT::i32,
/*   381*/   OPC_MoveParent,
/*   382*/   OPC_CheckPredicate, 0, // Predicate_shl_oneuse
/*   384*/   OPC_MoveParent,
/*   385*/   OPC_MoveChild1,
/*   386*/   OPC_CheckOpcode, TARGET_VAL(ISD::SUB),
/*   389*/   OPC_CheckChild0Integer, 32, 
/*   391*/   OPC_CheckChild1Same, 1,
/*   393*/   OPC_CheckType, MVT::i32,
/*   395*/   OPC_MoveParent,
/*   396*/   OPC_CheckType, MVT::i32,
/*   398*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*   400*/   OPC_EmitInteger, MVT::i32, 0, 
/*   403*/   OPC_EmitInteger, MVT::i32, 0, 
/*   406*/   OPC_EmitInteger, MVT::i32, 0, 
/*   409*/   OPC_EmitInteger, MVT::i32, 0, 
/*   412*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*   424*/   OPC_EmitInteger, MVT::i32, 0, 
/*   427*/   OPC_EmitNode1, TARGET_VAL(R600::MOV_IMM_I32), 0,
                 MVT::i32, 1/*#Ops*/, 7,  // Results = #8
/*   434*/   OPC_EmitInteger, MVT::i32, 0, 
/*   437*/   OPC_EmitInteger, MVT::i32, 0, 
/*   440*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*   452*/   OPC_EmitInteger, MVT::i32, 0, 
/*   455*/   OPC_EmitInteger, MVT::i32, 0, 
/*   458*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*   470*/   OPC_EmitInteger, MVT::i32, 1, 
/*   473*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*   476*/   OPC_EmitInteger, MVT::i32, 0, 
/*   479*/   OPC_EmitInteger, MVT::i32, 0, 
/*   482*/   OPC_MorphNodeTo1, TARGET_VAL(R600::BFE_INT_eg), 0,
                 MVT::i32, 18/*#Ops*/, 2, 3, 0, 4, 5, 6, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 
             // Src: (sra:{ *:[i32] } (shl:{ *:[i32] } i32:{ *:[i32] }:$src, (sub:{ *:[i32] } 32:{ *:[i32] }, i32:{ *:[i32] }:$width))<<P:Predicate_shl_oneuse>>, (sub:{ *:[i32] } 32:{ *:[i32] }, i32:{ *:[i32] }:$width)) - Complexity = 23
             // Dst: (BFE_INT_eg:{ *:[i32] } ?:{ *:[i32] }:$src, (MOV_IMM_I32:{ *:[i32] } 0:{ *:[i32] }), ?:{ *:[i32] }:$width)
/*   506*/  /*Scope*/ 82|128,1/*210*/, /*->718*/
/*   508*/   OPC_RecordChild0, // #0 = $src0
/*   509*/   OPC_RecordChild1, // #1 = $src1
/*   510*/   OPC_CheckChild1Type, MVT::i32,
/*   512*/   OPC_CheckType, MVT::i32,
/*   514*/   OPC_Scope, 100, /*->616*/ // 2 children in Scope
/*   516*/    OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/*   518*/    OPC_EmitInteger, MVT::i32, 0, 
/*   521*/    OPC_EmitInteger, MVT::i32, 0, 
/*   524*/    OPC_EmitInteger, MVT::i32, 1, 
/*   527*/    OPC_EmitInteger, MVT::i32, 0, 
/*   530*/    OPC_EmitInteger, MVT::i32, 0, 
/*   533*/    OPC_EmitInteger, MVT::i32, 0, 
/*   536*/    OPC_EmitInteger, MVT::i32, 0, 
/*   539*/    OPC_EmitInteger, MVT::i32, 0, 
/*   542*/    OPC_EmitInteger, MVT::i32, 0, 
/*   545*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*   557*/    OPC_EmitInteger, MVT::i32, 0, 
/*   560*/    OPC_EmitInteger, MVT::i32, 0, 
/*   563*/    OPC_EmitInteger, MVT::i32, 0, 
/*   566*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*   578*/    OPC_EmitInteger, MVT::i32, 1, 
/*   581*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*   584*/    OPC_EmitInteger, MVT::i32, 0, 
/*   587*/    OPC_EmitInteger, MVT::i32, 0, 
/*   590*/    OPC_MorphNodeTo1, TARGET_VAL(R600::ASHR_r600), 0,
                  MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
              // Src: (sra:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
              // Dst: (ASHR_r600:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/*   616*/   /*Scope*/ 100, /*->717*/
/*   617*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*   619*/    OPC_EmitInteger, MVT::i32, 0, 
/*   622*/    OPC_EmitInteger, MVT::i32, 0, 
/*   625*/    OPC_EmitInteger, MVT::i32, 1, 
/*   628*/    OPC_EmitInteger, MVT::i32, 0, 
/*   631*/    OPC_EmitInteger, MVT::i32, 0, 
/*   634*/    OPC_EmitInteger, MVT::i32, 0, 
/*   637*/    OPC_EmitInteger, MVT::i32, 0, 
/*   640*/    OPC_EmitInteger, MVT::i32, 0, 
/*   643*/    OPC_EmitInteger, MVT::i32, 0, 
/*   646*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*   658*/    OPC_EmitInteger, MVT::i32, 0, 
/*   661*/    OPC_EmitInteger, MVT::i32, 0, 
/*   664*/    OPC_EmitInteger, MVT::i32, 0, 
/*   667*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*   679*/    OPC_EmitInteger, MVT::i32, 1, 
/*   682*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*   685*/    OPC_EmitInteger, MVT::i32, 0, 
/*   688*/    OPC_EmitInteger, MVT::i32, 0, 
/*   691*/    OPC_MorphNodeTo1, TARGET_VAL(R600::ASHR_eg), 0,
                  MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
              // Src: (sra:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
              // Dst: (ASHR_eg:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/*   717*/   0, /*End of Scope*/
/*   718*/  0, /*End of Scope*/
/*   719*/ /*SwitchOpcode*/ 126, TARGET_VAL(AMDGPUISD::R600_EXPORT),// ->848
/*   722*/  OPC_RecordNode, // #0 = 'R600_EXPORT' chained node
/*   723*/  OPC_RecordChild1, // #1 = $src
/*   724*/  OPC_CheckChild1Type, MVT::v4f32,
/*   726*/  OPC_RecordChild2, // #2 = $base
/*   727*/  OPC_MoveChild2,
/*   728*/  OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*   731*/  OPC_CheckType, MVT::i32,
/*   733*/  OPC_MoveParent,
/*   734*/  OPC_RecordChild3, // #3 = $type
/*   735*/  OPC_MoveChild3,
/*   736*/  OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*   739*/  OPC_CheckType, MVT::i32,
/*   741*/  OPC_MoveParent,
/*   742*/  OPC_RecordChild4, // #4 = $swz_x
/*   743*/  OPC_MoveChild4,
/*   744*/  OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*   747*/  OPC_CheckType, MVT::i32,
/*   749*/  OPC_MoveParent,
/*   750*/  OPC_RecordChild5, // #5 = $swz_y
/*   751*/  OPC_MoveChild5,
/*   752*/  OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*   755*/  OPC_CheckType, MVT::i32,
/*   757*/  OPC_MoveParent,
/*   758*/  OPC_RecordChild6, // #6 = $swz_z
/*   759*/  OPC_MoveChild6,
/*   760*/  OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*   763*/  OPC_CheckType, MVT::i32,
/*   765*/  OPC_MoveParent,
/*   766*/  OPC_RecordChild7, // #7 = $swz_w
/*   767*/  OPC_MoveChild7,
/*   768*/  OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*   771*/  OPC_CheckType, MVT::i32,
/*   773*/  OPC_MoveParent,
/*   774*/  OPC_Scope, 35, /*->811*/ // 2 children in Scope
/*   776*/   OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/*   778*/   OPC_EmitMergeInputChains1_0,
/*   779*/   OPC_EmitConvertToTarget, 3,
/*   781*/   OPC_EmitConvertToTarget, 2,
/*   783*/   OPC_EmitConvertToTarget, 4,
/*   785*/   OPC_EmitConvertToTarget, 5,
/*   787*/   OPC_EmitConvertToTarget, 6,
/*   789*/   OPC_EmitConvertToTarget, 7,
/*   791*/   OPC_EmitInteger, MVT::i32, 39, 
/*   794*/   OPC_EmitInteger, MVT::i32, 0, 
/*   797*/   OPC_MorphNodeTo0, TARGET_VAL(R600::R600_ExportSwz), 0|OPFL_Chain,
                 9/*#Ops*/, 1, 8, 9, 10, 11, 12, 13, 14, 15, 
             // Src: (R600_EXPORT R600_Reg128:{ *:[v4f32] }:$src, (imm:{ *:[i32] }):$base, (imm:{ *:[i32] }):$type, (imm:{ *:[i32] }):$swz_x, (imm:{ *:[i32] }):$swz_y, (imm:{ *:[i32] }):$swz_z, (imm:{ *:[i32] }):$swz_w) - Complexity = 21
             // Dst: (R600_ExportSwz R600_Reg128:{ *:[v4f32] }:$src, (imm:{ *:[i32] }):$type, (imm:{ *:[i32] }):$base, (imm:{ *:[i32] }):$swz_x, (imm:{ *:[i32] }):$swz_y, (imm:{ *:[i32] }):$swz_z, (imm:{ *:[i32] }):$swz_w, 39:{ *:[i32] }, 0:{ *:[i32] })
/*   811*/  /*Scope*/ 35, /*->847*/
/*   812*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*   814*/   OPC_EmitMergeInputChains1_0,
/*   815*/   OPC_EmitConvertToTarget, 3,
/*   817*/   OPC_EmitConvertToTarget, 2,
/*   819*/   OPC_EmitConvertToTarget, 4,
/*   821*/   OPC_EmitConvertToTarget, 5,
/*   823*/   OPC_EmitConvertToTarget, 6,
/*   825*/   OPC_EmitConvertToTarget, 7,
/*   827*/   OPC_EmitInteger, MVT::i32, 83, 
/*   830*/   OPC_EmitInteger, MVT::i32, 0, 
/*   833*/   OPC_MorphNodeTo0, TARGET_VAL(R600::EG_ExportSwz), 0|OPFL_Chain,
                 9/*#Ops*/, 1, 8, 9, 10, 11, 12, 13, 14, 15, 
             // Src: (R600_EXPORT R600_Reg128:{ *:[v4f32] }:$src, (imm:{ *:[i32] }):$base, (imm:{ *:[i32] }):$type, (imm:{ *:[i32] }):$swz_x, (imm:{ *:[i32] }):$swz_y, (imm:{ *:[i32] }):$swz_z, (imm:{ *:[i32] }):$swz_w) - Complexity = 21
             // Dst: (EG_ExportSwz R600_Reg128:{ *:[v4f32] }:$src, (imm:{ *:[i32] }):$type, (imm:{ *:[i32] }):$base, (imm:{ *:[i32] }):$swz_x, (imm:{ *:[i32] }):$swz_y, (imm:{ *:[i32] }):$swz_z, (imm:{ *:[i32] }):$swz_w, 83:{ *:[i32] }, 0:{ *:[i32] })
/*   847*/  0, /*End of Scope*/
/*   848*/ /*SwitchOpcode*/ 84|128,8/*1108*/, TARGET_VAL(ISD::AND),// ->1960
/*   852*/  OPC_Scope, 37|128,2/*293*/, /*->1148*/ // 5 children in Scope
/*   855*/   OPC_RecordChild0, // #0 = $src
/*   856*/   OPC_MoveChild1,
/*   857*/   OPC_SwitchOpcode /*2 cases */, 8|128,1/*136*/, TARGET_VAL(ISD::ADD),// ->998
/*   862*/    OPC_MoveChild0,
/*   863*/    OPC_CheckOpcode, TARGET_VAL(ISD::SHL),
/*   866*/    OPC_CheckChild0Integer, 1, 
/*   868*/    OPC_RecordChild1, // #1 = $width
/*   869*/    OPC_CheckChild1Type, MVT::i32,
/*   871*/    OPC_CheckPredicate, 0, // Predicate_shl_oneuse
/*   873*/    OPC_MoveParent,
/*   874*/    OPC_CheckChild1Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*   885*/    OPC_CheckPredicate, 0, // Predicate_add_oneuse
/*   887*/    OPC_MoveParent,
/*   888*/    OPC_CheckType, MVT::i32,
/*   890*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*   892*/    OPC_EmitInteger, MVT::i32, 0, 
/*   895*/    OPC_EmitInteger, MVT::i32, 0, 
/*   898*/    OPC_EmitInteger, MVT::i32, 0, 
/*   901*/    OPC_EmitInteger, MVT::i32, 0, 
/*   904*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*   916*/    OPC_EmitInteger, MVT::i32, 0, 
/*   919*/    OPC_EmitNode1, TARGET_VAL(R600::MOV_IMM_I32), 0,
                  MVT::i32, 1/*#Ops*/, 7,  // Results = #8
/*   926*/    OPC_EmitInteger, MVT::i32, 0, 
/*   929*/    OPC_EmitInteger, MVT::i32, 0, 
/*   932*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*   944*/    OPC_EmitInteger, MVT::i32, 0, 
/*   947*/    OPC_EmitInteger, MVT::i32, 0, 
/*   950*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*   962*/    OPC_EmitInteger, MVT::i32, 1, 
/*   965*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*   968*/    OPC_EmitInteger, MVT::i32, 0, 
/*   971*/    OPC_EmitInteger, MVT::i32, 0, 
/*   974*/    OPC_MorphNodeTo1, TARGET_VAL(R600::BFE_UINT_eg), 0,
                  MVT::i32, 18/*#Ops*/, 2, 3, 0, 4, 5, 6, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 
              // Src: (and:{ *:[i32] } i32:{ *:[i32] }:$src, (add:{ *:[i32] } (shl:{ *:[i32] } 1:{ *:[i32] }, i32:{ *:[i32] }:$width)<<P:Predicate_shl_oneuse>>, -1:{ *:[i32] })<<P:Predicate_add_oneuse>>) - Complexity = 21
              // Dst: (BFE_UINT_eg:{ *:[i32] } ?:{ *:[i32] }:$src, (MOV_IMM_I32:{ *:[i32] } 0:{ *:[i32] }), ?:{ *:[i32] }:$width)
/*   998*/   /*SwitchOpcode*/ 17|128,1/*145*/, TARGET_VAL(ISD::XOR),// ->1147
/*  1002*/    OPC_MoveChild0,
/*  1003*/    OPC_CheckOpcode, TARGET_VAL(ISD::SHL),
/*  1006*/    OPC_CheckChild0Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1017*/    OPC_RecordChild1, // #1 = $width
/*  1018*/    OPC_CheckChild1Type, MVT::i32,
/*  1020*/    OPC_CheckPredicate, 0, // Predicate_shl_oneuse
/*  1022*/    OPC_MoveParent,
/*  1023*/    OPC_CheckChild1Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1034*/    OPC_CheckPredicate, 0, // Predicate_xor_oneuse
/*  1036*/    OPC_MoveParent,
/*  1037*/    OPC_CheckType, MVT::i32,
/*  1039*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  1041*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1044*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1047*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1050*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1053*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1065*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1068*/    OPC_EmitNode1, TARGET_VAL(R600::MOV_IMM_I32), 0,
                  MVT::i32, 1/*#Ops*/, 7,  // Results = #8
/*  1075*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1078*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1081*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1093*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1096*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1099*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1111*/    OPC_EmitInteger, MVT::i32, 1, 
/*  1114*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  1117*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1120*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1123*/    OPC_MorphNodeTo1, TARGET_VAL(R600::BFE_UINT_eg), 0,
                  MVT::i32, 18/*#Ops*/, 2, 3, 0, 4, 5, 6, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 
              // Src: (and:{ *:[i32] } i32:{ *:[i32] }:$src, (xor:{ *:[i32] } (shl:{ *:[i32] } -1:{ *:[i32] }, i32:{ *:[i32] }:$width)<<P:Predicate_shl_oneuse>>, -1:{ *:[i32] })<<P:Predicate_xor_oneuse>>) - Complexity = 21
              // Dst: (BFE_UINT_eg:{ *:[i32] } ?:{ *:[i32] }:$src, (MOV_IMM_I32:{ *:[i32] } 0:{ *:[i32] }), ?:{ *:[i32] }:$width)
/*  1147*/   0, // EndSwitchOpcode
/*  1148*/  /*Scope*/ 38|128,2/*294*/, /*->1444*/
/*  1150*/   OPC_MoveChild0,
/*  1151*/   OPC_SwitchOpcode /*2 cases */, 9|128,1/*137*/, TARGET_VAL(ISD::ADD),// ->1293
/*  1156*/    OPC_MoveChild0,
/*  1157*/    OPC_CheckOpcode, TARGET_VAL(ISD::SHL),
/*  1160*/    OPC_CheckChild0Integer, 1, 
/*  1162*/    OPC_RecordChild1, // #0 = $width
/*  1163*/    OPC_CheckChild1Type, MVT::i32,
/*  1165*/    OPC_CheckPredicate, 0, // Predicate_shl_oneuse
/*  1167*/    OPC_MoveParent,
/*  1168*/    OPC_CheckChild1Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1179*/    OPC_CheckPredicate, 0, // Predicate_add_oneuse
/*  1181*/    OPC_MoveParent,
/*  1182*/    OPC_RecordChild1, // #1 = $src
/*  1183*/    OPC_CheckType, MVT::i32,
/*  1185*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  1187*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1190*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1193*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1196*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1199*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1211*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1214*/    OPC_EmitNode1, TARGET_VAL(R600::MOV_IMM_I32), 0,
                  MVT::i32, 1/*#Ops*/, 7,  // Results = #8
/*  1221*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1224*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1227*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1239*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1242*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1245*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1257*/    OPC_EmitInteger, MVT::i32, 1, 
/*  1260*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  1263*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1266*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1269*/    OPC_MorphNodeTo1, TARGET_VAL(R600::BFE_UINT_eg), 0,
                  MVT::i32, 18/*#Ops*/, 2, 3, 1, 4, 5, 6, 8, 9, 10, 11, 0, 12, 13, 14, 15, 16, 17, 18, 
              // Src: (and:{ *:[i32] } (add:{ *:[i32] } (shl:{ *:[i32] } 1:{ *:[i32] }, i32:{ *:[i32] }:$width)<<P:Predicate_shl_oneuse>>, -1:{ *:[i32] })<<P:Predicate_add_oneuse>>, i32:{ *:[i32] }:$src) - Complexity = 21
              // Dst: (BFE_UINT_eg:{ *:[i32] } ?:{ *:[i32] }:$src, (MOV_IMM_I32:{ *:[i32] } 0:{ *:[i32] }), ?:{ *:[i32] }:$width)
/*  1293*/   /*SwitchOpcode*/ 18|128,1/*146*/, TARGET_VAL(ISD::XOR),// ->1443
/*  1297*/    OPC_MoveChild0,
/*  1298*/    OPC_CheckOpcode, TARGET_VAL(ISD::SHL),
/*  1301*/    OPC_CheckChild0Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1312*/    OPC_RecordChild1, // #0 = $width
/*  1313*/    OPC_CheckChild1Type, MVT::i32,
/*  1315*/    OPC_CheckPredicate, 0, // Predicate_shl_oneuse
/*  1317*/    OPC_MoveParent,
/*  1318*/    OPC_CheckChild1Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1329*/    OPC_CheckPredicate, 0, // Predicate_xor_oneuse
/*  1331*/    OPC_MoveParent,
/*  1332*/    OPC_RecordChild1, // #1 = $src
/*  1333*/    OPC_CheckType, MVT::i32,
/*  1335*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  1337*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1340*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1343*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1346*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1349*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1361*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1364*/    OPC_EmitNode1, TARGET_VAL(R600::MOV_IMM_I32), 0,
                  MVT::i32, 1/*#Ops*/, 7,  // Results = #8
/*  1371*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1374*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1377*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1389*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1392*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1395*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1407*/    OPC_EmitInteger, MVT::i32, 1, 
/*  1410*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  1413*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1416*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1419*/    OPC_MorphNodeTo1, TARGET_VAL(R600::BFE_UINT_eg), 0,
                  MVT::i32, 18/*#Ops*/, 2, 3, 1, 4, 5, 6, 8, 9, 10, 11, 0, 12, 13, 14, 15, 16, 17, 18, 
              // Src: (and:{ *:[i32] } (xor:{ *:[i32] } (shl:{ *:[i32] } -1:{ *:[i32] }, i32:{ *:[i32] }:$width)<<P:Predicate_shl_oneuse>>, -1:{ *:[i32] })<<P:Predicate_xor_oneuse>>, i32:{ *:[i32] }:$src) - Complexity = 21
              // Dst: (BFE_UINT_eg:{ *:[i32] } ?:{ *:[i32] }:$src, (MOV_IMM_I32:{ *:[i32] } 0:{ *:[i32] }), ?:{ *:[i32] }:$width)
/*  1443*/   0, // EndSwitchOpcode
/*  1444*/  /*Scope*/ 11|128,1/*139*/, /*->1585*/
/*  1446*/   OPC_RecordChild0, // #0 = $src
/*  1447*/   OPC_MoveChild1,
/*  1448*/   OPC_CheckOpcode, TARGET_VAL(ISD::SRL),
/*  1451*/   OPC_CheckChild0Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1462*/   OPC_MoveChild1,
/*  1463*/   OPC_CheckOpcode, TARGET_VAL(ISD::SUB),
/*  1466*/   OPC_CheckChild0Integer, 32, 
/*  1468*/   OPC_RecordChild1, // #1 = $width
/*  1469*/   OPC_CheckType, MVT::i32,
/*  1471*/   OPC_MoveParent,
/*  1472*/   OPC_CheckPredicate, 0, // Predicate_srl_oneuse
/*  1474*/   OPC_MoveParent,
/*  1475*/   OPC_CheckType, MVT::i32,
/*  1477*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  1479*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1482*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1485*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1488*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1491*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1503*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1506*/   OPC_EmitNode1, TARGET_VAL(R600::MOV_IMM_I32), 0,
                 MVT::i32, 1/*#Ops*/, 7,  // Results = #8
/*  1513*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1516*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1519*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1531*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1534*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1537*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1549*/   OPC_EmitInteger, MVT::i32, 1, 
/*  1552*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  1555*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1558*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1561*/   OPC_MorphNodeTo1, TARGET_VAL(R600::BFE_UINT_eg), 0,
                 MVT::i32, 18/*#Ops*/, 2, 3, 0, 4, 5, 6, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 
             // Src: (and:{ *:[i32] } i32:{ *:[i32] }:$src, (srl:{ *:[i32] } -1:{ *:[i32] }, (sub:{ *:[i32] } 32:{ *:[i32] }, i32:{ *:[i32] }:$width))<<P:Predicate_srl_oneuse>>) - Complexity = 20
             // Dst: (BFE_UINT_eg:{ *:[i32] } ?:{ *:[i32] }:$src, (MOV_IMM_I32:{ *:[i32] } 0:{ *:[i32] }), ?:{ *:[i32] }:$width)
/*  1585*/  /*Scope*/ 11|128,2/*267*/, /*->1854*/
/*  1587*/   OPC_MoveChild0,
/*  1588*/   OPC_CheckOpcode, TARGET_VAL(ISD::SRL),
/*  1591*/   OPC_Scope, 7|128,1/*135*/, /*->1729*/ // 2 children in Scope
/*  1594*/    OPC_CheckChild0Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1605*/    OPC_MoveChild1,
/*  1606*/    OPC_CheckOpcode, TARGET_VAL(ISD::SUB),
/*  1609*/    OPC_CheckChild0Integer, 32, 
/*  1611*/    OPC_RecordChild1, // #0 = $width
/*  1612*/    OPC_CheckType, MVT::i32,
/*  1614*/    OPC_MoveParent,
/*  1615*/    OPC_CheckPredicate, 0, // Predicate_srl_oneuse
/*  1617*/    OPC_MoveParent,
/*  1618*/    OPC_RecordChild1, // #1 = $src
/*  1619*/    OPC_CheckType, MVT::i32,
/*  1621*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  1623*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1626*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1629*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1632*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1635*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1647*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1650*/    OPC_EmitNode1, TARGET_VAL(R600::MOV_IMM_I32), 0,
                  MVT::i32, 1/*#Ops*/, 7,  // Results = #8
/*  1657*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1660*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1663*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1675*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1678*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1681*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1693*/    OPC_EmitInteger, MVT::i32, 1, 
/*  1696*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  1699*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1702*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1705*/    OPC_MorphNodeTo1, TARGET_VAL(R600::BFE_UINT_eg), 0,
                  MVT::i32, 18/*#Ops*/, 2, 3, 1, 4, 5, 6, 8, 9, 10, 11, 0, 12, 13, 14, 15, 16, 17, 18, 
              // Src: (and:{ *:[i32] } (srl:{ *:[i32] } -1:{ *:[i32] }, (sub:{ *:[i32] } 32:{ *:[i32] }, i32:{ *:[i32] }:$width))<<P:Predicate_srl_oneuse>>, i32:{ *:[i32] }:$src) - Complexity = 20
              // Dst: (BFE_UINT_eg:{ *:[i32] } ?:{ *:[i32] }:$src, (MOV_IMM_I32:{ *:[i32] } 0:{ *:[i32] }), ?:{ *:[i32] }:$width)
/*  1729*/   /*Scope*/ 123, /*->1853*/
/*  1730*/    OPC_RecordChild0, // #0 = $src
/*  1731*/    OPC_RecordChild1, // #1 = $rshift
/*  1732*/    OPC_CheckChild1Type, MVT::i32,
/*  1734*/    OPC_MoveParent,
/*  1735*/    OPC_RecordChild1, // #2 = $mask
/*  1736*/    OPC_MoveChild1,
/*  1737*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  1740*/    OPC_CheckPredicate, 1, // Predicate_IMMZeroBasedBitfieldMask
/*  1742*/    OPC_MoveParent,
/*  1743*/    OPC_CheckType, MVT::i32,
/*  1745*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  1747*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1750*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1753*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1756*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1759*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1771*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1774*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1777*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1789*/    OPC_EmitNodeXForm, 0, 2, // IMMPopCount
/*  1792*/    OPC_EmitNode1, TARGET_VAL(R600::MOV_IMM_I32), 0,
                  MVT::i32, 1/*#Ops*/, 11,  // Results = #12
/*  1799*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1802*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1805*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1817*/    OPC_EmitInteger, MVT::i32, 1, 
/*  1820*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  1823*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1826*/    OPC_EmitInteger, MVT::i32, 0, 
/*  1829*/    OPC_MorphNodeTo1, TARGET_VAL(R600::BFE_UINT_eg), 0,
                  MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 
              // Src: (and:{ *:[i32] } (srl:{ *:[i32] } i32:{ *:[i32] }:$src, i32:{ *:[i32] }:$rshift), (imm:{ *:[i32] })<<P:Predicate_IMMZeroBasedBitfieldMask>>:$mask) - Complexity = 10
              // Dst: (BFE_UINT_eg:{ *:[i32] } ?:{ *:[i32] }:$src, ?:{ *:[i32] }:$rshift, (MOV_IMM_I32:{ *:[i32] } (IMMPopCount:{ *:[i32] } ?:{ *:[i32] }:$mask)))
/*  1853*/   0, /*End of Scope*/
/*  1854*/  /*Scope*/ 104, /*->1959*/
/*  1855*/   OPC_RecordChild0, // #0 = $src0
/*  1856*/   OPC_RecordChild1, // #1 = $src1
/*  1857*/   OPC_CheckType, MVT::i32,
/*  1859*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  1861*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1864*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1867*/   OPC_EmitInteger, MVT::i32, 1, 
/*  1870*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1873*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1876*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1879*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1882*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1885*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1888*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1900*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1903*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1906*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1909*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  1921*/   OPC_EmitInteger, MVT::i32, 1, 
/*  1924*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  1927*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1930*/   OPC_EmitInteger, MVT::i32, 0, 
/*  1933*/   OPC_MorphNodeTo1, TARGET_VAL(R600::AND_INT), 0,
                 MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
             // Src: (and:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
             // Dst: (AND_INT:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/*  1959*/  0, /*End of Scope*/
/*  1960*/ /*SwitchOpcode*/ 118|128,3/*502*/, TARGET_VAL(ISD::INTRINSIC_VOID),// ->2466
/*  1964*/  OPC_RecordNode, // #0 = 'intrinsic_void' chained node
/*  1965*/  OPC_Scope, 64|128,2/*320*/, /*->2288*/ // 4 children in Scope
/*  1968*/   OPC_CheckChild1Integer, 123|128,50/*6523*/, 
/*  1971*/   OPC_RecordChild2, // #1 = $src
/*  1972*/   OPC_RecordChild3, // #2 = $arraybase
/*  1973*/   OPC_MoveChild3,
/*  1974*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  1977*/   OPC_MoveParent,
/*  1978*/   OPC_Scope, 76, /*->2056*/ // 4 children in Scope
/*  1980*/    OPC_CheckChild4Integer, 0, 
/*  1982*/    OPC_RecordChild5, // #3 = $mask
/*  1983*/    OPC_MoveChild5,
/*  1984*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  1987*/    OPC_MoveParent,
/*  1988*/    OPC_Scope, 32, /*->2022*/ // 2 children in Scope
/*  1990*/     OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/*  1992*/     OPC_EmitMergeInputChains1_0,
/*  1993*/     OPC_EmitInteger, MVT::i32, 0, 
/*  1996*/     OPC_EmitConvertToTarget, 2,
/*  1998*/     OPC_EmitInteger, MVT::i32, 127|128,31/*4095*/, 
/*  2002*/     OPC_EmitConvertToTarget, 3,
/*  2004*/     OPC_EmitInteger, MVT::i32, 32, 
/*  2007*/     OPC_EmitInteger, MVT::i32, 0, 
/*  2010*/     OPC_MorphNodeTo0, TARGET_VAL(R600::R600_ExportBuf), 0|OPFL_Chain,
                   7/*#Ops*/, 1, 4, 5, 6, 7, 8, 9, 
               // Src: (intrinsic_void 6523:{ *:[iPTR] }, R600_Reg128:{ *:[v4f32] }:$src, (imm:{ *:[i32] }):$arraybase, 0:{ *:[i32] }, (imm:{ *:[i32] }):$mask) - Complexity = 19
               // Dst: (R600_ExportBuf R600_Reg128:{ *:[v4f32] }:$src, 0:{ *:[i32] }, (imm:{ *:[i32] }):$arraybase, 4095:{ *:[i32] }, (imm:{ *:[i32] }):$mask, 32:{ *:[i32] }, 0:{ *:[i32] })
/*  2022*/    /*Scope*/ 32, /*->2055*/
/*  2023*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  2025*/     OPC_EmitMergeInputChains1_0,
/*  2026*/     OPC_EmitInteger, MVT::i32, 0, 
/*  2029*/     OPC_EmitConvertToTarget, 2,
/*  2031*/     OPC_EmitInteger, MVT::i32, 127|128,31/*4095*/, 
/*  2035*/     OPC_EmitConvertToTarget, 3,
/*  2037*/     OPC_EmitInteger, MVT::i32, 64, 
/*  2040*/     OPC_EmitInteger, MVT::i32, 0, 
/*  2043*/     OPC_MorphNodeTo0, TARGET_VAL(R600::EG_ExportBuf), 0|OPFL_Chain,
                   7/*#Ops*/, 1, 4, 5, 6, 7, 8, 9, 
               // Src: (intrinsic_void 6523:{ *:[iPTR] }, R600_Reg128:{ *:[v4f32] }:$src, (imm:{ *:[i32] }):$arraybase, 0:{ *:[i32] }, (imm:{ *:[i32] }):$mask) - Complexity = 19
               // Dst: (EG_ExportBuf R600_Reg128:{ *:[v4f32] }:$src, 0:{ *:[i32] }, (imm:{ *:[i32] }):$arraybase, 4095:{ *:[i32] }, (imm:{ *:[i32] }):$mask, 64:{ *:[i32] }, 0:{ *:[i32] })
/*  2055*/    0, /*End of Scope*/
/*  2056*/   /*Scope*/ 76, /*->2133*/
/*  2057*/    OPC_CheckChild4Integer, 1, 
/*  2059*/    OPC_RecordChild5, // #3 = $mask
/*  2060*/    OPC_MoveChild5,
/*  2061*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2064*/    OPC_MoveParent,
/*  2065*/    OPC_Scope, 32, /*->2099*/ // 2 children in Scope
/*  2067*/     OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/*  2069*/     OPC_EmitMergeInputChains1_0,
/*  2070*/     OPC_EmitInteger, MVT::i32, 0, 
/*  2073*/     OPC_EmitConvertToTarget, 2,
/*  2075*/     OPC_EmitInteger, MVT::i32, 127|128,31/*4095*/, 
/*  2079*/     OPC_EmitConvertToTarget, 3,
/*  2081*/     OPC_EmitInteger, MVT::i32, 33, 
/*  2084*/     OPC_EmitInteger, MVT::i32, 0, 
/*  2087*/     OPC_MorphNodeTo0, TARGET_VAL(R600::R600_ExportBuf), 0|OPFL_Chain,
                   7/*#Ops*/, 1, 4, 5, 6, 7, 8, 9, 
               // Src: (intrinsic_void 6523:{ *:[iPTR] }, R600_Reg128:{ *:[v4f32] }:$src, (imm:{ *:[i32] }):$arraybase, 1:{ *:[i32] }, (imm:{ *:[i32] }):$mask) - Complexity = 19
               // Dst: (R600_ExportBuf ?:{ *:[v4f32] }:$src, 0:{ *:[i32] }, (imm:{ *:[i32] }):$arraybase, 4095:{ *:[i32] }, (imm:{ *:[i32] }):$mask, 33:{ *:[i32] }, 0:{ *:[i32] })
/*  2099*/    /*Scope*/ 32, /*->2132*/
/*  2100*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  2102*/     OPC_EmitMergeInputChains1_0,
/*  2103*/     OPC_EmitInteger, MVT::i32, 0, 
/*  2106*/     OPC_EmitConvertToTarget, 2,
/*  2108*/     OPC_EmitInteger, MVT::i32, 127|128,31/*4095*/, 
/*  2112*/     OPC_EmitConvertToTarget, 3,
/*  2114*/     OPC_EmitInteger, MVT::i32, 65, 
/*  2117*/     OPC_EmitInteger, MVT::i32, 0, 
/*  2120*/     OPC_MorphNodeTo0, TARGET_VAL(R600::EG_ExportBuf), 0|OPFL_Chain,
                   7/*#Ops*/, 1, 4, 5, 6, 7, 8, 9, 
               // Src: (intrinsic_void 6523:{ *:[iPTR] }, R600_Reg128:{ *:[v4f32] }:$src, (imm:{ *:[i32] }):$arraybase, 1:{ *:[i32] }, (imm:{ *:[i32] }):$mask) - Complexity = 19
               // Dst: (EG_ExportBuf ?:{ *:[v4f32] }:$src, 0:{ *:[i32] }, (imm:{ *:[i32] }):$arraybase, 4095:{ *:[i32] }, (imm:{ *:[i32] }):$mask, 65:{ *:[i32] }, 0:{ *:[i32] })
/*  2132*/    0, /*End of Scope*/
/*  2133*/   /*Scope*/ 76, /*->2210*/
/*  2134*/    OPC_CheckChild4Integer, 2, 
/*  2136*/    OPC_RecordChild5, // #3 = $mask
/*  2137*/    OPC_MoveChild5,
/*  2138*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2141*/    OPC_MoveParent,
/*  2142*/    OPC_Scope, 32, /*->2176*/ // 2 children in Scope
/*  2144*/     OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/*  2146*/     OPC_EmitMergeInputChains1_0,
/*  2147*/     OPC_EmitInteger, MVT::i32, 0, 
/*  2150*/     OPC_EmitConvertToTarget, 2,
/*  2152*/     OPC_EmitInteger, MVT::i32, 127|128,31/*4095*/, 
/*  2156*/     OPC_EmitConvertToTarget, 3,
/*  2158*/     OPC_EmitInteger, MVT::i32, 34, 
/*  2161*/     OPC_EmitInteger, MVT::i32, 0, 
/*  2164*/     OPC_MorphNodeTo0, TARGET_VAL(R600::R600_ExportBuf), 0|OPFL_Chain,
                   7/*#Ops*/, 1, 4, 5, 6, 7, 8, 9, 
               // Src: (intrinsic_void 6523:{ *:[iPTR] }, R600_Reg128:{ *:[v4f32] }:$src, (imm:{ *:[i32] }):$arraybase, 2:{ *:[i32] }, (imm:{ *:[i32] }):$mask) - Complexity = 19
               // Dst: (R600_ExportBuf ?:{ *:[v4f32] }:$src, 0:{ *:[i32] }, (imm:{ *:[i32] }):$arraybase, 4095:{ *:[i32] }, (imm:{ *:[i32] }):$mask, 34:{ *:[i32] }, 0:{ *:[i32] })
/*  2176*/    /*Scope*/ 32, /*->2209*/
/*  2177*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  2179*/     OPC_EmitMergeInputChains1_0,
/*  2180*/     OPC_EmitInteger, MVT::i32, 0, 
/*  2183*/     OPC_EmitConvertToTarget, 2,
/*  2185*/     OPC_EmitInteger, MVT::i32, 127|128,31/*4095*/, 
/*  2189*/     OPC_EmitConvertToTarget, 3,
/*  2191*/     OPC_EmitInteger, MVT::i32, 66, 
/*  2194*/     OPC_EmitInteger, MVT::i32, 0, 
/*  2197*/     OPC_MorphNodeTo0, TARGET_VAL(R600::EG_ExportBuf), 0|OPFL_Chain,
                   7/*#Ops*/, 1, 4, 5, 6, 7, 8, 9, 
               // Src: (intrinsic_void 6523:{ *:[iPTR] }, R600_Reg128:{ *:[v4f32] }:$src, (imm:{ *:[i32] }):$arraybase, 2:{ *:[i32] }, (imm:{ *:[i32] }):$mask) - Complexity = 19
               // Dst: (EG_ExportBuf ?:{ *:[v4f32] }:$src, 0:{ *:[i32] }, (imm:{ *:[i32] }):$arraybase, 4095:{ *:[i32] }, (imm:{ *:[i32] }):$mask, 66:{ *:[i32] }, 0:{ *:[i32] })
/*  2209*/    0, /*End of Scope*/
/*  2210*/   /*Scope*/ 76, /*->2287*/
/*  2211*/    OPC_CheckChild4Integer, 3, 
/*  2213*/    OPC_RecordChild5, // #3 = $mask
/*  2214*/    OPC_MoveChild5,
/*  2215*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2218*/    OPC_MoveParent,
/*  2219*/    OPC_Scope, 32, /*->2253*/ // 2 children in Scope
/*  2221*/     OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/*  2223*/     OPC_EmitMergeInputChains1_0,
/*  2224*/     OPC_EmitInteger, MVT::i32, 0, 
/*  2227*/     OPC_EmitConvertToTarget, 2,
/*  2229*/     OPC_EmitInteger, MVT::i32, 127|128,31/*4095*/, 
/*  2233*/     OPC_EmitConvertToTarget, 3,
/*  2235*/     OPC_EmitInteger, MVT::i32, 35, 
/*  2238*/     OPC_EmitInteger, MVT::i32, 0, 
/*  2241*/     OPC_MorphNodeTo0, TARGET_VAL(R600::R600_ExportBuf), 0|OPFL_Chain,
                   7/*#Ops*/, 1, 4, 5, 6, 7, 8, 9, 
               // Src: (intrinsic_void 6523:{ *:[iPTR] }, R600_Reg128:{ *:[v4f32] }:$src, (imm:{ *:[i32] }):$arraybase, 3:{ *:[i32] }, (imm:{ *:[i32] }):$mask) - Complexity = 19
               // Dst: (R600_ExportBuf ?:{ *:[v4f32] }:$src, 0:{ *:[i32] }, (imm:{ *:[i32] }):$arraybase, 4095:{ *:[i32] }, (imm:{ *:[i32] }):$mask, 35:{ *:[i32] }, 0:{ *:[i32] })
/*  2253*/    /*Scope*/ 32, /*->2286*/
/*  2254*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  2256*/     OPC_EmitMergeInputChains1_0,
/*  2257*/     OPC_EmitInteger, MVT::i32, 0, 
/*  2260*/     OPC_EmitConvertToTarget, 2,
/*  2262*/     OPC_EmitInteger, MVT::i32, 127|128,31/*4095*/, 
/*  2266*/     OPC_EmitConvertToTarget, 3,
/*  2268*/     OPC_EmitInteger, MVT::i32, 67, 
/*  2271*/     OPC_EmitInteger, MVT::i32, 0, 
/*  2274*/     OPC_MorphNodeTo0, TARGET_VAL(R600::EG_ExportBuf), 0|OPFL_Chain,
                   7/*#Ops*/, 1, 4, 5, 6, 7, 8, 9, 
               // Src: (intrinsic_void 6523:{ *:[iPTR] }, R600_Reg128:{ *:[v4f32] }:$src, (imm:{ *:[i32] }):$arraybase, 3:{ *:[i32] }, (imm:{ *:[i32] }):$mask) - Complexity = 19
               // Dst: (EG_ExportBuf ?:{ *:[v4f32] }:$src, 0:{ *:[i32] }, (imm:{ *:[i32] }):$arraybase, 4095:{ *:[i32] }, (imm:{ *:[i32] }):$mask, 67:{ *:[i32] }, 0:{ *:[i32] })
/*  2286*/    0, /*End of Scope*/
/*  2287*/   0, /*End of Scope*/
/*  2288*/  /*Scope*/ 49, /*->2338*/
/*  2289*/   OPC_CheckChild1Integer, 105|128,50/*6505*/, 
/*  2292*/   OPC_RecordChild2, // #1 = $rw_gpr
/*  2293*/   OPC_RecordChild3, // #2 = $index_gpr
/*  2294*/   OPC_RecordChild4, // #3 = $rat_id
/*  2295*/   OPC_MoveChild4,
/*  2296*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2299*/   OPC_MoveParent,
/*  2300*/   OPC_Scope, 17, /*->2319*/ // 2 children in Scope
/*  2302*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/*  2304*/    OPC_EmitMergeInputChains1_0,
/*  2305*/    OPC_EmitConvertToTarget, 3,
/*  2307*/    OPC_EmitInteger, MVT::i32, 0, 
/*  2310*/    OPC_MorphNodeTo0, TARGET_VAL(R600::RAT_STORE_TYPED_cm), 0|OPFL_Chain,
                  4/*#Ops*/, 1, 2, 4, 5, 
              // Src: (intrinsic_void 6505:{ *:[iPTR] }, R600_Reg128:{ *:[v4i32] }:$rw_gpr, R600_Reg128:{ *:[v4i32] }:$index_gpr, (imm:{ *:[i32] }):$rat_id) - Complexity = 11
              // Dst: (RAT_STORE_TYPED_cm R600_Reg128:{ *:[v4i32] }:$rw_gpr, R600_Reg128:{ *:[v4i32] }:$index_gpr, (imm:{ *:[i32] }):$rat_id)
/*  2319*/   /*Scope*/ 17, /*->2337*/
/*  2320*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  2322*/    OPC_EmitMergeInputChains1_0,
/*  2323*/    OPC_EmitConvertToTarget, 3,
/*  2325*/    OPC_EmitInteger, MVT::i32, 0, 
/*  2328*/    OPC_MorphNodeTo0, TARGET_VAL(R600::RAT_STORE_TYPED_eg), 0|OPFL_Chain,
                  4/*#Ops*/, 1, 2, 4, 5, 
              // Src: (intrinsic_void 6505:{ *:[iPTR] }, R600_Reg128:{ *:[v4i32] }:$rw_gpr, R600_Reg128:{ *:[v4i32] }:$index_gpr, (imm:{ *:[i32] }):$rat_id) - Complexity = 11
              // Dst: (RAT_STORE_TYPED_eg R600_Reg128:{ *:[v4i32] }:$rw_gpr, R600_Reg128:{ *:[v4i32] }:$index_gpr, (imm:{ *:[i32] }):$rat_id)
/*  2337*/   0, /*End of Scope*/
/*  2338*/  /*Scope*/ 11, /*->2350*/
/*  2339*/   OPC_CheckChild1Integer, 102|128,50/*6502*/, 
/*  2342*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  2344*/   OPC_EmitMergeInputChains1_0,
/*  2345*/   OPC_MorphNodeTo0, TARGET_VAL(R600::GROUP_BARRIER), 0|OPFL_Chain,
                 0/*#Ops*/, 
             // Src: (intrinsic_void 6502:{ *:[iPTR] }) - Complexity = 8
             // Dst: (GROUP_BARRIER)
/*  2350*/  /*Scope*/ 114, /*->2465*/
/*  2351*/   OPC_CheckChild1Integer, 104|128,50/*6504*/, 
/*  2354*/   OPC_RecordChild2, // #1 = $src0
/*  2355*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  2357*/   OPC_EmitMergeInputChains1_0,
/*  2358*/   OPC_EmitInteger, MVT::i32, 0, 
/*  2361*/   OPC_EmitInteger, MVT::i32, 0, 
/*  2364*/   OPC_EmitInteger, MVT::i32, 1, 
/*  2367*/   OPC_EmitInteger, MVT::i32, 0, 
/*  2370*/   OPC_EmitInteger, MVT::i32, 0, 
/*  2373*/   OPC_EmitInteger, MVT::i32, 0, 
/*  2376*/   OPC_EmitRegister, MVT::f32, R600::ZERO,
/*  2379*/   OPC_EmitInteger, MVT::i32, 0, 
/*  2382*/   OPC_EmitInteger, MVT::i32, 0, 
/*  2385*/   OPC_EmitInteger, MVT::i32, 0, 
/*  2388*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  2400*/   OPC_EmitInteger, MVT::i32, 0, 
/*  2403*/   OPC_EmitInteger, MVT::i32, 0, 
/*  2406*/   OPC_EmitInteger, MVT::i32, 0, 
/*  2409*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  2421*/   OPC_EmitInteger, MVT::i32, 1, 
/*  2424*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  2427*/   OPC_EmitInteger, MVT::i32, 0, 
/*  2430*/   OPC_EmitInteger, MVT::i32, 0, 
/*  2433*/   OPC_EmitNode1, TARGET_VAL(R600::KILLGT), 0|OPFL_Chain,
                 MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 13, 14, 15, 16, 17, 18, 19, 20,  // Results = #21
/*  2459*/   OPC_MorphNodeTo0, TARGET_VAL(R600::MASK_WRITE), 0|OPFL_Chain,
                 1/*#Ops*/, 21, 
             // Src: (intrinsic_void 6504:{ *:[iPTR] }, f32:{ *:[f32] }:$src0) - Complexity = 8
             // Dst: (MASK_WRITE (KILLGT:{ *:[i32] } ZERO:{ *:[f32] }, ?:{ *:[f32] }:$src0))
/*  2465*/  0, /*End of Scope*/
/*  2466*/ /*SwitchOpcode*/ 15|128,116/*14863*/, TARGET_VAL(ISD::OR),// ->17333
/*  2470*/  OPC_Scope, 34|128,115/*14754*/, /*->17227*/ // 2 children in Scope
/*  2473*/   OPC_MoveChild0,
/*  2474*/   OPC_CheckOpcode, TARGET_VAL(ISD::AND),
/*  2477*/   OPC_Scope, 93|128,5/*733*/, /*->3213*/ // 8 children in Scope
/*  2480*/    OPC_RecordChild0, // #0 = $y
/*  2481*/    OPC_Scope, 108|128,3/*492*/, /*->2976*/ // 2 children in Scope
/*  2484*/     OPC_RecordChild1, // #1 = $x
/*  2485*/     OPC_MoveParent,
/*  2486*/     OPC_MoveChild1,
/*  2487*/     OPC_CheckOpcode, TARGET_VAL(ISD::AND),
/*  2490*/     OPC_Scope, 120, /*->2612*/ // 4 children in Scope
/*  2492*/      OPC_RecordChild0, // #2 = $z
/*  2493*/      OPC_MoveChild1,
/*  2494*/      OPC_CheckOpcode, TARGET_VAL(ISD::XOR),
/*  2497*/      OPC_CheckChild0Same, 1,
/*  2499*/      OPC_CheckChild1Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  2510*/      OPC_MoveParent,
/*  2511*/      OPC_MoveParent,
/*  2512*/      OPC_CheckType, MVT::i32,
/*  2514*/      OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  2516*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2519*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2522*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2525*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2528*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  2540*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2543*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2546*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  2558*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2561*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2564*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  2576*/      OPC_EmitInteger, MVT::i32, 1, 
/*  2579*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  2582*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2585*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2588*/      OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 3, 4, 1, 5, 6, 7, 0, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
                // Src: (or:{ *:[i32] } (and:{ *:[i32] } i32:{ *:[i32] }:$y, i32:{ *:[i32] }:$x), (and:{ *:[i32] } i32:{ *:[i32] }:$z, (xor:{ *:[i32] } i32:{ *:[i32] }:$x, -1:{ *:[i32] }))) - Complexity = 17
                // Dst: (BFI_INT_eg:{ *:[i32] } ?:{ *:[i32] }:$x, ?:{ *:[i32] }:$y, ?:{ *:[i32] }:$z)
/*  2612*/     /*Scope*/ 120, /*->2733*/
/*  2613*/      OPC_MoveChild0,
/*  2614*/      OPC_CheckOpcode, TARGET_VAL(ISD::XOR),
/*  2617*/      OPC_CheckChild0Same, 1,
/*  2619*/      OPC_CheckChild1Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  2630*/      OPC_MoveParent,
/*  2631*/      OPC_RecordChild1, // #2 = $z
/*  2632*/      OPC_MoveParent,
/*  2633*/      OPC_CheckType, MVT::i32,
/*  2635*/      OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  2637*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2640*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2643*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2646*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2649*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  2661*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2664*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2667*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  2679*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2682*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2685*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  2697*/      OPC_EmitInteger, MVT::i32, 1, 
/*  2700*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  2703*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2706*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2709*/      OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 3, 4, 1, 5, 6, 7, 0, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
                // Src: (or:{ *:[i32] } (and:{ *:[i32] } i32:{ *:[i32] }:$y, i32:{ *:[i32] }:$x), (and:{ *:[i32] } (xor:{ *:[i32] } i32:{ *:[i32] }:$x, -1:{ *:[i32] }), i32:{ *:[i32] }:$z)) - Complexity = 17
                // Dst: (BFI_INT_eg:{ *:[i32] } ?:{ *:[i32] }:$x, ?:{ *:[i32] }:$y, ?:{ *:[i32] }:$z)
/*  2733*/     /*Scope*/ 120, /*->2854*/
/*  2734*/      OPC_RecordChild0, // #2 = $z
/*  2735*/      OPC_MoveChild1,
/*  2736*/      OPC_CheckOpcode, TARGET_VAL(ISD::XOR),
/*  2739*/      OPC_CheckChild0Same, 0,
/*  2741*/      OPC_CheckChild1Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  2752*/      OPC_MoveParent,
/*  2753*/      OPC_MoveParent,
/*  2754*/      OPC_CheckType, MVT::i32,
/*  2756*/      OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  2758*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2761*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2764*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2767*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2770*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  2782*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2785*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2788*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  2800*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2803*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2806*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  2818*/      OPC_EmitInteger, MVT::i32, 1, 
/*  2821*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  2824*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2827*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2830*/      OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
                // Src: (or:{ *:[i32] } (and:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$y), (and:{ *:[i32] } i32:{ *:[i32] }:$z, (xor:{ *:[i32] } i32:{ *:[i32] }:$x, -1:{ *:[i32] }))) - Complexity = 17
                // Dst: (BFI_INT_eg:{ *:[i32] } ?:{ *:[i32] }:$x, ?:{ *:[i32] }:$y, ?:{ *:[i32] }:$z)
/*  2854*/     /*Scope*/ 120, /*->2975*/
/*  2855*/      OPC_MoveChild0,
/*  2856*/      OPC_CheckOpcode, TARGET_VAL(ISD::XOR),
/*  2859*/      OPC_CheckChild0Same, 0,
/*  2861*/      OPC_CheckChild1Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  2872*/      OPC_MoveParent,
/*  2873*/      OPC_RecordChild1, // #2 = $z
/*  2874*/      OPC_MoveParent,
/*  2875*/      OPC_CheckType, MVT::i32,
/*  2877*/      OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  2879*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2882*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2885*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2888*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2891*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  2903*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2906*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2909*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  2921*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2924*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2927*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  2939*/      OPC_EmitInteger, MVT::i32, 1, 
/*  2942*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  2945*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2948*/      OPC_EmitInteger, MVT::i32, 0, 
/*  2951*/      OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
                // Src: (or:{ *:[i32] } (and:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$y), (and:{ *:[i32] } (xor:{ *:[i32] } i32:{ *:[i32] }:$x, -1:{ *:[i32] }), i32:{ *:[i32] }:$z)) - Complexity = 17
                // Dst: (BFI_INT_eg:{ *:[i32] } ?:{ *:[i32] }:$x, ?:{ *:[i32] }:$y, ?:{ *:[i32] }:$z)
/*  2975*/     0, /*End of Scope*/
/*  2976*/    /*Scope*/ 106|128,1/*234*/, /*->3212*/
/*  2978*/     OPC_MoveChild1,
/*  2979*/     OPC_CheckOpcode, TARGET_VAL(ISD::XOR),
/*  2982*/     OPC_RecordChild0, // #1 = $x
/*  2983*/     OPC_CheckChild1Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  2994*/     OPC_MoveParent,
/*  2995*/     OPC_MoveParent,
/*  2996*/     OPC_MoveChild1,
/*  2997*/     OPC_CheckOpcode, TARGET_VAL(ISD::AND),
/*  3000*/     OPC_Scope, 104, /*->3106*/ // 2 children in Scope
/*  3002*/      OPC_RecordChild0, // #2 = $y
/*  3003*/      OPC_CheckChild1Same, 1,
/*  3005*/      OPC_MoveParent,
/*  3006*/      OPC_CheckType, MVT::i32,
/*  3008*/      OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  3010*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3013*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3016*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3019*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3022*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3034*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3037*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3040*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3052*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3055*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3058*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3070*/      OPC_EmitInteger, MVT::i32, 1, 
/*  3073*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  3076*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3079*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3082*/      OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 3, 4, 1, 5, 6, 7, 2, 8, 9, 10, 0, 11, 12, 13, 14, 15, 16, 17, 
                // Src: (or:{ *:[i32] } (and:{ *:[i32] } i32:{ *:[i32] }:$z, (xor:{ *:[i32] } i32:{ *:[i32] }:$x, -1:{ *:[i32] })), (and:{ *:[i32] } i32:{ *:[i32] }:$y, i32:{ *:[i32] }:$x)) - Complexity = 17
                // Dst: (BFI_INT_eg:{ *:[i32] } ?:{ *:[i32] }:$x, ?:{ *:[i32] }:$y, ?:{ *:[i32] }:$z)
/*  3106*/     /*Scope*/ 104, /*->3211*/
/*  3107*/      OPC_CheckChild0Same, 1,
/*  3109*/      OPC_RecordChild1, // #2 = $y
/*  3110*/      OPC_MoveParent,
/*  3111*/      OPC_CheckType, MVT::i32,
/*  3113*/      OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  3115*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3118*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3121*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3124*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3127*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3139*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3142*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3145*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3157*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3160*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3163*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3175*/      OPC_EmitInteger, MVT::i32, 1, 
/*  3178*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  3181*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3184*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3187*/      OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 3, 4, 1, 5, 6, 7, 2, 8, 9, 10, 0, 11, 12, 13, 14, 15, 16, 17, 
                // Src: (or:{ *:[i32] } (and:{ *:[i32] } i32:{ *:[i32] }:$z, (xor:{ *:[i32] } i32:{ *:[i32] }:$x, -1:{ *:[i32] })), (and:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$y)) - Complexity = 17
                // Dst: (BFI_INT_eg:{ *:[i32] } ?:{ *:[i32] }:$x, ?:{ *:[i32] }:$y, ?:{ *:[i32] }:$z)
/*  3211*/     0, /*End of Scope*/
/*  3212*/    0, /*End of Scope*/
/*  3213*/   /*Scope*/ 107|128,1/*235*/, /*->3450*/
/*  3215*/    OPC_MoveChild0,
/*  3216*/    OPC_CheckOpcode, TARGET_VAL(ISD::XOR),
/*  3219*/    OPC_RecordChild0, // #0 = $x
/*  3220*/    OPC_CheckChild1Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3231*/    OPC_MoveParent,
/*  3232*/    OPC_RecordChild1, // #1 = $z
/*  3233*/    OPC_MoveParent,
/*  3234*/    OPC_MoveChild1,
/*  3235*/    OPC_CheckOpcode, TARGET_VAL(ISD::AND),
/*  3238*/    OPC_Scope, 104, /*->3344*/ // 2 children in Scope
/*  3240*/     OPC_RecordChild0, // #2 = $y
/*  3241*/     OPC_CheckChild1Same, 0,
/*  3243*/     OPC_MoveParent,
/*  3244*/     OPC_CheckType, MVT::i32,
/*  3246*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  3248*/     OPC_EmitInteger, MVT::i32, 0, 
/*  3251*/     OPC_EmitInteger, MVT::i32, 0, 
/*  3254*/     OPC_EmitInteger, MVT::i32, 0, 
/*  3257*/     OPC_EmitInteger, MVT::i32, 0, 
/*  3260*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3272*/     OPC_EmitInteger, MVT::i32, 0, 
/*  3275*/     OPC_EmitInteger, MVT::i32, 0, 
/*  3278*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3290*/     OPC_EmitInteger, MVT::i32, 0, 
/*  3293*/     OPC_EmitInteger, MVT::i32, 0, 
/*  3296*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3308*/     OPC_EmitInteger, MVT::i32, 1, 
/*  3311*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  3314*/     OPC_EmitInteger, MVT::i32, 0, 
/*  3317*/     OPC_EmitInteger, MVT::i32, 0, 
/*  3320*/     OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 2, 8, 9, 10, 1, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (or:{ *:[i32] } (and:{ *:[i32] } (xor:{ *:[i32] } i32:{ *:[i32] }:$x, -1:{ *:[i32] }), i32:{ *:[i32] }:$z), (and:{ *:[i32] } i32:{ *:[i32] }:$y, i32:{ *:[i32] }:$x)) - Complexity = 17
               // Dst: (BFI_INT_eg:{ *:[i32] } ?:{ *:[i32] }:$x, ?:{ *:[i32] }:$y, ?:{ *:[i32] }:$z)
/*  3344*/    /*Scope*/ 104, /*->3449*/
/*  3345*/     OPC_CheckChild0Same, 0,
/*  3347*/     OPC_RecordChild1, // #2 = $y
/*  3348*/     OPC_MoveParent,
/*  3349*/     OPC_CheckType, MVT::i32,
/*  3351*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  3353*/     OPC_EmitInteger, MVT::i32, 0, 
/*  3356*/     OPC_EmitInteger, MVT::i32, 0, 
/*  3359*/     OPC_EmitInteger, MVT::i32, 0, 
/*  3362*/     OPC_EmitInteger, MVT::i32, 0, 
/*  3365*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3377*/     OPC_EmitInteger, MVT::i32, 0, 
/*  3380*/     OPC_EmitInteger, MVT::i32, 0, 
/*  3383*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3395*/     OPC_EmitInteger, MVT::i32, 0, 
/*  3398*/     OPC_EmitInteger, MVT::i32, 0, 
/*  3401*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3413*/     OPC_EmitInteger, MVT::i32, 1, 
/*  3416*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  3419*/     OPC_EmitInteger, MVT::i32, 0, 
/*  3422*/     OPC_EmitInteger, MVT::i32, 0, 
/*  3425*/     OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 2, 8, 9, 10, 1, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (or:{ *:[i32] } (and:{ *:[i32] } (xor:{ *:[i32] } i32:{ *:[i32] }:$x, -1:{ *:[i32] }), i32:{ *:[i32] }:$z), (and:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$y)) - Complexity = 17
               // Dst: (BFI_INT_eg:{ *:[i32] } ?:{ *:[i32] }:$x, ?:{ *:[i32] }:$y, ?:{ *:[i32] }:$z)
/*  3449*/    0, /*End of Scope*/
/*  3450*/   /*Scope*/ 39|128,14/*1831*/, /*->5283*/
/*  3452*/    OPC_RecordChild0, // #0 = $y
/*  3453*/    OPC_Scope, 72|128,9/*1224*/, /*->4680*/ // 2 children in Scope
/*  3456*/     OPC_RecordChild1, // #1 = $x
/*  3457*/     OPC_MoveParent,
/*  3458*/     OPC_MoveChild1,
/*  3459*/     OPC_CheckOpcode, TARGET_VAL(ISD::AND),
/*  3462*/     OPC_Scope, 46|128,2/*302*/, /*->3767*/ // 4 children in Scope
/*  3465*/      OPC_RecordChild0, // #2 = $z
/*  3466*/      OPC_MoveChild1,
/*  3467*/      OPC_CheckOpcode, TARGET_VAL(ISD::XOR),
/*  3470*/      OPC_CheckChild0Same, 1,
/*  3472*/      OPC_CheckChild1Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3483*/      OPC_MoveParent,
/*  3484*/      OPC_MoveParent,
/*  3485*/      OPC_CheckType, MVT::i64,
/*  3487*/      OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  3489*/      OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/*  3492*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3495*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3498*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  3501*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 1, 6,  // Results = #7
/*  3509*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3512*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3515*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3527*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  3530*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 11,  // Results = #12
/*  3538*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3541*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3544*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3556*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  3559*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 16,  // Results = #17
/*  3567*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3570*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3573*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3585*/      OPC_EmitInteger, MVT::i32, 1, 
/*  3588*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  3591*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3594*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3597*/      OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24,  // Results = #25
/*  3621*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  3624*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3627*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3630*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  3633*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 1, 29,  // Results = #30
/*  3641*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3644*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3647*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3659*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  3662*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 34,  // Results = #35
/*  3670*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3673*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3676*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3688*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  3691*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 39,  // Results = #40
/*  3699*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3702*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3705*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3717*/      OPC_EmitInteger, MVT::i32, 1, 
/*  3720*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  3723*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3726*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3729*/      OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 27, 28, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47,  // Results = #48
/*  3753*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  3756*/      OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                    MVT::i64, 5/*#Ops*/, 3, 25, 26, 48, 49, 
                // Src: (or:{ *:[i64] } (and:{ *:[i64] } i64:{ *:[i64] }:$y, i64:{ *:[i64] }:$x), (and:{ *:[i64] } i64:{ *:[i64] }:$z, (xor:{ *:[i64] } i64:{ *:[i64] }:$x, -1:{ *:[i64] }))) - Complexity = 17
                // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/*  3767*/     /*Scope*/ 46|128,2/*302*/, /*->4071*/
/*  3769*/      OPC_MoveChild0,
/*  3770*/      OPC_CheckOpcode, TARGET_VAL(ISD::XOR),
/*  3773*/      OPC_CheckChild0Same, 1,
/*  3775*/      OPC_CheckChild1Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3786*/      OPC_MoveParent,
/*  3787*/      OPC_RecordChild1, // #2 = $z
/*  3788*/      OPC_MoveParent,
/*  3789*/      OPC_CheckType, MVT::i64,
/*  3791*/      OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  3793*/      OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/*  3796*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3799*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3802*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  3805*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 1, 6,  // Results = #7
/*  3813*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3816*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3819*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3831*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  3834*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 11,  // Results = #12
/*  3842*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3845*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3848*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3860*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  3863*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 16,  // Results = #17
/*  3871*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3874*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3877*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3889*/      OPC_EmitInteger, MVT::i32, 1, 
/*  3892*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  3895*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3898*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3901*/      OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24,  // Results = #25
/*  3925*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  3928*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3931*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3934*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  3937*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 1, 29,  // Results = #30
/*  3945*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3948*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3951*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3963*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  3966*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 34,  // Results = #35
/*  3974*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3977*/      OPC_EmitInteger, MVT::i32, 0, 
/*  3980*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  3992*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  3995*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 39,  // Results = #40
/*  4003*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4006*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4009*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4021*/      OPC_EmitInteger, MVT::i32, 1, 
/*  4024*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  4027*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4030*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4033*/      OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 27, 28, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47,  // Results = #48
/*  4057*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  4060*/      OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                    MVT::i64, 5/*#Ops*/, 3, 25, 26, 48, 49, 
                // Src: (or:{ *:[i64] } (and:{ *:[i64] } i64:{ *:[i64] }:$y, i64:{ *:[i64] }:$x), (and:{ *:[i64] } (xor:{ *:[i64] } i64:{ *:[i64] }:$x, -1:{ *:[i64] }), i64:{ *:[i64] }:$z)) - Complexity = 17
                // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/*  4071*/     /*Scope*/ 46|128,2/*302*/, /*->4375*/
/*  4073*/      OPC_RecordChild0, // #2 = $z
/*  4074*/      OPC_MoveChild1,
/*  4075*/      OPC_CheckOpcode, TARGET_VAL(ISD::XOR),
/*  4078*/      OPC_CheckChild0Same, 0,
/*  4080*/      OPC_CheckChild1Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4091*/      OPC_MoveParent,
/*  4092*/      OPC_MoveParent,
/*  4093*/      OPC_CheckType, MVT::i64,
/*  4095*/      OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  4097*/      OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/*  4100*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4103*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4106*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  4109*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 6,  // Results = #7
/*  4117*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4120*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4123*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4135*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  4138*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 1, 11,  // Results = #12
/*  4146*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4149*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4152*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4164*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  4167*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 16,  // Results = #17
/*  4175*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4178*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4181*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4193*/      OPC_EmitInteger, MVT::i32, 1, 
/*  4196*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  4199*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4202*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4205*/      OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24,  // Results = #25
/*  4229*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  4232*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4235*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4238*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  4241*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 29,  // Results = #30
/*  4249*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4252*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4255*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4267*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  4270*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 1, 34,  // Results = #35
/*  4278*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4281*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4284*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4296*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  4299*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 39,  // Results = #40
/*  4307*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4310*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4313*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4325*/      OPC_EmitInteger, MVT::i32, 1, 
/*  4328*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  4331*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4334*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4337*/      OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 27, 28, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47,  // Results = #48
/*  4361*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  4364*/      OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                    MVT::i64, 5/*#Ops*/, 3, 25, 26, 48, 49, 
                // Src: (or:{ *:[i64] } (and:{ *:[i64] } i64:{ *:[i64] }:$x, i64:{ *:[i64] }:$y), (and:{ *:[i64] } i64:{ *:[i64] }:$z, (xor:{ *:[i64] } i64:{ *:[i64] }:$x, -1:{ *:[i64] }))) - Complexity = 17
                // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/*  4375*/     /*Scope*/ 46|128,2/*302*/, /*->4679*/
/*  4377*/      OPC_MoveChild0,
/*  4378*/      OPC_CheckOpcode, TARGET_VAL(ISD::XOR),
/*  4381*/      OPC_CheckChild0Same, 0,
/*  4383*/      OPC_CheckChild1Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4394*/      OPC_MoveParent,
/*  4395*/      OPC_RecordChild1, // #2 = $z
/*  4396*/      OPC_MoveParent,
/*  4397*/      OPC_CheckType, MVT::i64,
/*  4399*/      OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  4401*/      OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/*  4404*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4407*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4410*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  4413*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 6,  // Results = #7
/*  4421*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4424*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4427*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4439*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  4442*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 1, 11,  // Results = #12
/*  4450*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4453*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4456*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4468*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  4471*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 16,  // Results = #17
/*  4479*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4482*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4485*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4497*/      OPC_EmitInteger, MVT::i32, 1, 
/*  4500*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  4503*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4506*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4509*/      OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24,  // Results = #25
/*  4533*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  4536*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4539*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4542*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  4545*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 29,  // Results = #30
/*  4553*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4556*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4559*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4571*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  4574*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 1, 34,  // Results = #35
/*  4582*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4585*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4588*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4600*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  4603*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 39,  // Results = #40
/*  4611*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4614*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4617*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4629*/      OPC_EmitInteger, MVT::i32, 1, 
/*  4632*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  4635*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4638*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4641*/      OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 27, 28, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47,  // Results = #48
/*  4665*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  4668*/      OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                    MVT::i64, 5/*#Ops*/, 3, 25, 26, 48, 49, 
                // Src: (or:{ *:[i64] } (and:{ *:[i64] } i64:{ *:[i64] }:$x, i64:{ *:[i64] }:$y), (and:{ *:[i64] } (xor:{ *:[i64] } i64:{ *:[i64] }:$x, -1:{ *:[i64] }), i64:{ *:[i64] }:$z)) - Complexity = 17
                // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/*  4679*/     0, /*End of Scope*/
/*  4680*/    /*Scope*/ 88|128,4/*600*/, /*->5282*/
/*  4682*/     OPC_MoveChild1,
/*  4683*/     OPC_CheckOpcode, TARGET_VAL(ISD::XOR),
/*  4686*/     OPC_RecordChild0, // #1 = $x
/*  4687*/     OPC_CheckChild1Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4698*/     OPC_MoveParent,
/*  4699*/     OPC_MoveParent,
/*  4700*/     OPC_MoveChild1,
/*  4701*/     OPC_CheckOpcode, TARGET_VAL(ISD::AND),
/*  4704*/     OPC_Scope, 30|128,2/*286*/, /*->4993*/ // 2 children in Scope
/*  4707*/      OPC_RecordChild0, // #2 = $y
/*  4708*/      OPC_CheckChild1Same, 1,
/*  4710*/      OPC_MoveParent,
/*  4711*/      OPC_CheckType, MVT::i64,
/*  4713*/      OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  4715*/      OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/*  4718*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4721*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4724*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  4727*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 1, 6,  // Results = #7
/*  4735*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4738*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4741*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4753*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  4756*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 11,  // Results = #12
/*  4764*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4767*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4770*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4782*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  4785*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 16,  // Results = #17
/*  4793*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4796*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4799*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4811*/      OPC_EmitInteger, MVT::i32, 1, 
/*  4814*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  4817*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4820*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4823*/      OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24,  // Results = #25
/*  4847*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  4850*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4853*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4856*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  4859*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 1, 29,  // Results = #30
/*  4867*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4870*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4873*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4885*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  4888*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 34,  // Results = #35
/*  4896*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4899*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4902*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4914*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  4917*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 39,  // Results = #40
/*  4925*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4928*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4931*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  4943*/      OPC_EmitInteger, MVT::i32, 1, 
/*  4946*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  4949*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4952*/      OPC_EmitInteger, MVT::i32, 0, 
/*  4955*/      OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 27, 28, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47,  // Results = #48
/*  4979*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  4982*/      OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                    MVT::i64, 5/*#Ops*/, 3, 25, 26, 48, 49, 
                // Src: (or:{ *:[i64] } (and:{ *:[i64] } i64:{ *:[i64] }:$z, (xor:{ *:[i64] } i64:{ *:[i64] }:$x, -1:{ *:[i64] })), (and:{ *:[i64] } i64:{ *:[i64] }:$y, i64:{ *:[i64] }:$x)) - Complexity = 17
                // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/*  4993*/     /*Scope*/ 30|128,2/*286*/, /*->5281*/
/*  4995*/      OPC_CheckChild0Same, 1,
/*  4997*/      OPC_RecordChild1, // #2 = $y
/*  4998*/      OPC_MoveParent,
/*  4999*/      OPC_CheckType, MVT::i64,
/*  5001*/      OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  5003*/      OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/*  5006*/      OPC_EmitInteger, MVT::i32, 0, 
/*  5009*/      OPC_EmitInteger, MVT::i32, 0, 
/*  5012*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  5015*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 1, 6,  // Results = #7
/*  5023*/      OPC_EmitInteger, MVT::i32, 0, 
/*  5026*/      OPC_EmitInteger, MVT::i32, 0, 
/*  5029*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  5041*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  5044*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 11,  // Results = #12
/*  5052*/      OPC_EmitInteger, MVT::i32, 0, 
/*  5055*/      OPC_EmitInteger, MVT::i32, 0, 
/*  5058*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  5070*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  5073*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 16,  // Results = #17
/*  5081*/      OPC_EmitInteger, MVT::i32, 0, 
/*  5084*/      OPC_EmitInteger, MVT::i32, 0, 
/*  5087*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  5099*/      OPC_EmitInteger, MVT::i32, 1, 
/*  5102*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  5105*/      OPC_EmitInteger, MVT::i32, 0, 
/*  5108*/      OPC_EmitInteger, MVT::i32, 0, 
/*  5111*/      OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24,  // Results = #25
/*  5135*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/*  5138*/      OPC_EmitInteger, MVT::i32, 0, 
/*  5141*/      OPC_EmitInteger, MVT::i32, 0, 
/*  5144*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  5147*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 1, 29,  // Results = #30
/*  5155*/      OPC_EmitInteger, MVT::i32, 0, 
/*  5158*/      OPC_EmitInteger, MVT::i32, 0, 
/*  5161*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  5173*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  5176*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 34,  // Results = #35
/*  5184*/      OPC_EmitInteger, MVT::i32, 0, 
/*  5187*/      OPC_EmitInteger, MVT::i32, 0, 
/*  5190*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  5202*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  5205*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 39,  // Results = #40
/*  5213*/      OPC_EmitInteger, MVT::i32, 0, 
/*  5216*/      OPC_EmitInteger, MVT::i32, 0, 
/*  5219*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  5231*/      OPC_EmitInteger, MVT::i32, 1, 
/*  5234*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  5237*/      OPC_EmitInteger, MVT::i32, 0, 
/*  5240*/      OPC_EmitInteger, MVT::i32, 0, 
/*  5243*/      OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 27, 28, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47,  // Results = #48
/*  5267*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/*  5270*/      OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                    MVT::i64, 5/*#Ops*/, 3, 25, 26, 48, 49, 
                // Src: (or:{ *:[i64] } (and:{ *:[i64] } i64:{ *:[i64] }:$z, (xor:{ *:[i64] } i64:{ *:[i64] }:$x, -1:{ *:[i64] })), (and:{ *:[i64] } i64:{ *:[i64] }:$x, i64:{ *:[i64] }:$y)) - Complexity = 17
                // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/*  5281*/     0, /*End of Scope*/
/*  5282*/    0, /*End of Scope*/
/*  5283*/   /*Scope*/ 89|128,4/*601*/, /*->5886*/
/*  5285*/    OPC_MoveChild0,
/*  5286*/    OPC_CheckOpcode, TARGET_VAL(ISD::XOR),
/*  5289*/    OPC_RecordChild0, // #0 = $x
/*  5290*/    OPC_CheckChild1Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  5301*/    OPC_MoveParent,
/*  5302*/    OPC_RecordChild1, // #1 = $z
/*  5303*/    OPC_MoveParent,
/*  5304*/    OPC_MoveChild1,
/*  5305*/    OPC_CheckOpcode, TARGET_VAL(ISD::AND),
/*  5308*/    OPC_Scope, 30|128,2/*286*/, /*->5597*/ // 2 children in Scope
/*  5311*/     OPC_RecordChild0, // #2 = $y
/*  5312*/     OPC_CheckChild1Same, 0,
/*  5314*/     OPC_MoveParent,
/*  5315*/     OPC_CheckType, MVT::i64,
/*  5317*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  5319*/     OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/*  5322*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5325*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5328*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/*  5331*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 6,  // Results = #7
/*  5339*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5342*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5345*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  5357*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/*  5360*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 11,  // Results = #12
/*  5368*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5371*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5374*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  5386*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/*  5389*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 16,  // Results = #17
/*  5397*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5400*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5403*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  5415*/     OPC_EmitInteger, MVT::i32, 1, 
/*  5418*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  5421*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5424*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5427*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24,  // Results = #25
/*  5451*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/*  5454*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5457*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5460*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/*  5463*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 29,  // Results = #30
/*  5471*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5474*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5477*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  5489*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/*  5492*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 34,  // Results = #35
/*  5500*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5503*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5506*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  5518*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/*  5521*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 39,  // Results = #40
/*  5529*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5532*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5535*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  5547*/     OPC_EmitInteger, MVT::i32, 1, 
/*  5550*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  5553*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5556*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5559*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 27, 28, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47,  // Results = #48
/*  5583*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/*  5586*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                   MVT::i64, 5/*#Ops*/, 3, 25, 26, 48, 49, 
               // Src: (or:{ *:[i64] } (and:{ *:[i64] } (xor:{ *:[i64] } i64:{ *:[i64] }:$x, -1:{ *:[i64] }), i64:{ *:[i64] }:$z), (and:{ *:[i64] } i64:{ *:[i64] }:$y, i64:{ *:[i64] }:$x)) - Complexity = 17
               // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/*  5597*/    /*Scope*/ 30|128,2/*286*/, /*->5885*/
/*  5599*/     OPC_CheckChild0Same, 0,
/*  5601*/     OPC_RecordChild1, // #2 = $y
/*  5602*/     OPC_MoveParent,
/*  5603*/     OPC_CheckType, MVT::i64,
/*  5605*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  5607*/     OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/*  5610*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5613*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5616*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/*  5619*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 6,  // Results = #7
/*  5627*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5630*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5633*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  5645*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/*  5648*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 11,  // Results = #12
/*  5656*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5659*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5662*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  5674*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/*  5677*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 16,  // Results = #17
/*  5685*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5688*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5691*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  5703*/     OPC_EmitInteger, MVT::i32, 1, 
/*  5706*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  5709*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5712*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5715*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24,  // Results = #25
/*  5739*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/*  5742*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5745*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5748*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/*  5751*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 29,  // Results = #30
/*  5759*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5762*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5765*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  5777*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/*  5780*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 34,  // Results = #35
/*  5788*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5791*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5794*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  5806*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/*  5809*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 39,  // Results = #40
/*  5817*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5820*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5823*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  5835*/     OPC_EmitInteger, MVT::i32, 1, 
/*  5838*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  5841*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5844*/     OPC_EmitInteger, MVT::i32, 0, 
/*  5847*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 27, 28, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47,  // Results = #48
/*  5871*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/*  5874*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                   MVT::i64, 5/*#Ops*/, 3, 25, 26, 48, 49, 
               // Src: (or:{ *:[i64] } (and:{ *:[i64] } (xor:{ *:[i64] } i64:{ *:[i64] }:$x, -1:{ *:[i64] }), i64:{ *:[i64] }:$z), (and:{ *:[i64] } i64:{ *:[i64] }:$x, i64:{ *:[i64] }:$y)) - Complexity = 17
               // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/*  5885*/    0, /*End of Scope*/
/*  5886*/   /*Scope*/ 18|128,19/*2450*/, /*->8338*/
/*  5888*/    OPC_RecordChild0, // #0 = $x
/*  5889*/    OPC_Scope, 30|128,13/*1694*/, /*->7586*/ // 2 children in Scope
/*  5892*/     OPC_RecordChild1, // #1 = $z
/*  5893*/     OPC_MoveParent,
/*  5894*/     OPC_MoveChild1,
/*  5895*/     OPC_CheckOpcode, TARGET_VAL(ISD::AND),
/*  5898*/     OPC_Scope, 35|128,3/*419*/, /*->6320*/ // 4 children in Scope
/*  5901*/      OPC_RecordChild0, // #2 = $y
/*  5902*/      OPC_MoveChild1,
/*  5903*/      OPC_CheckOpcode, TARGET_VAL(ISD::OR),
/*  5906*/      OPC_Scope, 76|128,1/*204*/, /*->6113*/ // 2 children in Scope
/*  5909*/       OPC_CheckChild0Same, 0,
/*  5911*/       OPC_CheckChild1Same, 1,
/*  5913*/       OPC_MoveParent,
/*  5914*/       OPC_MoveParent,
/*  5915*/       OPC_CheckType, MVT::i32,
/*  5917*/       OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  5919*/       OPC_EmitInteger, MVT::i32, 0, 
/*  5922*/       OPC_EmitInteger, MVT::i32, 0, 
/*  5925*/       OPC_EmitInteger, MVT::i32, 0, 
/*  5928*/       OPC_EmitInteger, MVT::i32, 0, 
/*  5931*/       OPC_EmitInteger, MVT::i32, 1, 
/*  5934*/       OPC_EmitInteger, MVT::i32, 0, 
/*  5937*/       OPC_EmitInteger, MVT::i32, 0, 
/*  5940*/       OPC_EmitInteger, MVT::i32, 0, 
/*  5943*/       OPC_EmitInteger, MVT::i32, 0, 
/*  5946*/       OPC_EmitInteger, MVT::i32, 0, 
/*  5949*/       OPC_EmitInteger, MVT::i32, 0, 
/*  5952*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  5964*/       OPC_EmitInteger, MVT::i32, 0, 
/*  5967*/       OPC_EmitInteger, MVT::i32, 0, 
/*  5970*/       OPC_EmitInteger, MVT::i32, 0, 
/*  5973*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  5985*/       OPC_EmitInteger, MVT::i32, 1, 
/*  5988*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  5991*/       OPC_EmitInteger, MVT::i32, 0, 
/*  5994*/       OPC_EmitInteger, MVT::i32, 0, 
/*  5997*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 5, 6, 7, 8, 9, 10, 0, 11, 12, 13, 14, 2, 15, 16, 17, 18, 19, 20, 21, 22,  // Results = #23
/*  6023*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6026*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6029*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6041*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6044*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6047*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6059*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6062*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6065*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6077*/       OPC_EmitInteger, MVT::i32, 1, 
/*  6080*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  6083*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6086*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6089*/       OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 3, 4, 23, 24, 25, 26, 1, 27, 28, 29, 2, 30, 31, 32, 33, 34, 35, 36, 
                 // Src: (or:{ *:[i32] } (and:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$z), (and:{ *:[i32] } i32:{ *:[i32] }:$y, (or:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$z))) - Complexity = 12
                 // Dst: (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$y), i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$y)
/*  6113*/      /*Scope*/ 76|128,1/*204*/, /*->6319*/
/*  6115*/       OPC_CheckChild0Same, 1,
/*  6117*/       OPC_CheckChild1Same, 0,
/*  6119*/       OPC_MoveParent,
/*  6120*/       OPC_MoveParent,
/*  6121*/       OPC_CheckType, MVT::i32,
/*  6123*/       OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  6125*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6128*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6131*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6134*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6137*/       OPC_EmitInteger, MVT::i32, 1, 
/*  6140*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6143*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6146*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6149*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6152*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6155*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6158*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6170*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6173*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6176*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6179*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6191*/       OPC_EmitInteger, MVT::i32, 1, 
/*  6194*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  6197*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6200*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6203*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 5, 6, 7, 8, 9, 10, 0, 11, 12, 13, 14, 2, 15, 16, 17, 18, 19, 20, 21, 22,  // Results = #23
/*  6229*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6232*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6235*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6247*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6250*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6253*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6265*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6268*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6271*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6283*/       OPC_EmitInteger, MVT::i32, 1, 
/*  6286*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  6289*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6292*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6295*/       OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 3, 4, 23, 24, 25, 26, 1, 27, 28, 29, 2, 30, 31, 32, 33, 34, 35, 36, 
                 // Src: (or:{ *:[i32] } (and:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$z), (and:{ *:[i32] } i32:{ *:[i32] }:$y, (or:{ *:[i32] } i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$x))) - Complexity = 12
                 // Dst: (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$y), i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$y)
/*  6319*/      0, /*End of Scope*/
/*  6320*/     /*Scope*/ 36|128,3/*420*/, /*->6742*/
/*  6322*/      OPC_MoveChild0,
/*  6323*/      OPC_CheckOpcode, TARGET_VAL(ISD::OR),
/*  6326*/      OPC_Scope, 77|128,1/*205*/, /*->6534*/ // 2 children in Scope
/*  6329*/       OPC_CheckChild0Same, 0,
/*  6331*/       OPC_CheckChild1Same, 1,
/*  6333*/       OPC_MoveParent,
/*  6334*/       OPC_RecordChild1, // #2 = $y
/*  6335*/       OPC_MoveParent,
/*  6336*/       OPC_CheckType, MVT::i32,
/*  6338*/       OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  6340*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6343*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6346*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6349*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6352*/       OPC_EmitInteger, MVT::i32, 1, 
/*  6355*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6358*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6361*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6364*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6367*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6370*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6373*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6385*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6388*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6391*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6394*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6406*/       OPC_EmitInteger, MVT::i32, 1, 
/*  6409*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  6412*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6415*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6418*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 5, 6, 7, 8, 9, 10, 0, 11, 12, 13, 14, 2, 15, 16, 17, 18, 19, 20, 21, 22,  // Results = #23
/*  6444*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6447*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6450*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6462*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6465*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6468*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6480*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6483*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6486*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6498*/       OPC_EmitInteger, MVT::i32, 1, 
/*  6501*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  6504*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6507*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6510*/       OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 3, 4, 23, 24, 25, 26, 1, 27, 28, 29, 2, 30, 31, 32, 33, 34, 35, 36, 
                 // Src: (or:{ *:[i32] } (and:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$z), (and:{ *:[i32] } (or:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$z), i32:{ *:[i32] }:$y)) - Complexity = 12
                 // Dst: (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$y), i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$y)
/*  6534*/      /*Scope*/ 77|128,1/*205*/, /*->6741*/
/*  6536*/       OPC_CheckChild0Same, 1,
/*  6538*/       OPC_CheckChild1Same, 0,
/*  6540*/       OPC_MoveParent,
/*  6541*/       OPC_RecordChild1, // #2 = $y
/*  6542*/       OPC_MoveParent,
/*  6543*/       OPC_CheckType, MVT::i32,
/*  6545*/       OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  6547*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6550*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6553*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6556*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6559*/       OPC_EmitInteger, MVT::i32, 1, 
/*  6562*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6565*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6568*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6571*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6574*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6577*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6580*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6592*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6595*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6598*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6601*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6613*/       OPC_EmitInteger, MVT::i32, 1, 
/*  6616*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  6619*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6622*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6625*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 5, 6, 7, 8, 9, 10, 0, 11, 12, 13, 14, 2, 15, 16, 17, 18, 19, 20, 21, 22,  // Results = #23
/*  6651*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6654*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6657*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6669*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6672*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6675*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6687*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6690*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6693*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6705*/       OPC_EmitInteger, MVT::i32, 1, 
/*  6708*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  6711*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6714*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6717*/       OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 3, 4, 23, 24, 25, 26, 1, 27, 28, 29, 2, 30, 31, 32, 33, 34, 35, 36, 
                 // Src: (or:{ *:[i32] } (and:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$z), (and:{ *:[i32] } (or:{ *:[i32] } i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$x), i32:{ *:[i32] }:$y)) - Complexity = 12
                 // Dst: (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$y), i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$y)
/*  6741*/      0, /*End of Scope*/
/*  6742*/     /*Scope*/ 35|128,3/*419*/, /*->7163*/
/*  6744*/      OPC_RecordChild0, // #2 = $y
/*  6745*/      OPC_MoveChild1,
/*  6746*/      OPC_CheckOpcode, TARGET_VAL(ISD::OR),
/*  6749*/      OPC_Scope, 76|128,1/*204*/, /*->6956*/ // 2 children in Scope
/*  6752*/       OPC_CheckChild0Same, 1,
/*  6754*/       OPC_CheckChild1Same, 0,
/*  6756*/       OPC_MoveParent,
/*  6757*/       OPC_MoveParent,
/*  6758*/       OPC_CheckType, MVT::i32,
/*  6760*/       OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  6762*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6765*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6768*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6771*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6774*/       OPC_EmitInteger, MVT::i32, 1, 
/*  6777*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6780*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6783*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6786*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6789*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6792*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6795*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6807*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6810*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6813*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6816*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6828*/       OPC_EmitInteger, MVT::i32, 1, 
/*  6831*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  6834*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6837*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6840*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 5, 6, 7, 8, 9, 10, 1, 11, 12, 13, 14, 2, 15, 16, 17, 18, 19, 20, 21, 22,  // Results = #23
/*  6866*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6869*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6872*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6884*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6887*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6890*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6902*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6905*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6908*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  6920*/       OPC_EmitInteger, MVT::i32, 1, 
/*  6923*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  6926*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6929*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6932*/       OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 3, 4, 23, 24, 25, 26, 0, 27, 28, 29, 2, 30, 31, 32, 33, 34, 35, 36, 
                 // Src: (or:{ *:[i32] } (and:{ *:[i32] } i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$x), (and:{ *:[i32] } i32:{ *:[i32] }:$y, (or:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$z))) - Complexity = 12
                 // Dst: (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$y), i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$y)
/*  6956*/      /*Scope*/ 76|128,1/*204*/, /*->7162*/
/*  6958*/       OPC_CheckChild0Same, 0,
/*  6960*/       OPC_CheckChild1Same, 1,
/*  6962*/       OPC_MoveParent,
/*  6963*/       OPC_MoveParent,
/*  6964*/       OPC_CheckType, MVT::i32,
/*  6966*/       OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  6968*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6971*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6974*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6977*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6980*/       OPC_EmitInteger, MVT::i32, 1, 
/*  6983*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6986*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6989*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6992*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6995*/       OPC_EmitInteger, MVT::i32, 0, 
/*  6998*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7001*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7013*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7016*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7019*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7022*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7034*/       OPC_EmitInteger, MVT::i32, 1, 
/*  7037*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  7040*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7043*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7046*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 5, 6, 7, 8, 9, 10, 1, 11, 12, 13, 14, 2, 15, 16, 17, 18, 19, 20, 21, 22,  // Results = #23
/*  7072*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7075*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7078*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7090*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7093*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7096*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7108*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7111*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7114*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7126*/       OPC_EmitInteger, MVT::i32, 1, 
/*  7129*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  7132*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7135*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7138*/       OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 3, 4, 23, 24, 25, 26, 0, 27, 28, 29, 2, 30, 31, 32, 33, 34, 35, 36, 
                 // Src: (or:{ *:[i32] } (and:{ *:[i32] } i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$x), (and:{ *:[i32] } i32:{ *:[i32] }:$y, (or:{ *:[i32] } i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$x))) - Complexity = 12
                 // Dst: (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$y), i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$y)
/*  7162*/      0, /*End of Scope*/
/*  7163*/     /*Scope*/ 36|128,3/*420*/, /*->7585*/
/*  7165*/      OPC_MoveChild0,
/*  7166*/      OPC_CheckOpcode, TARGET_VAL(ISD::OR),
/*  7169*/      OPC_Scope, 77|128,1/*205*/, /*->7377*/ // 2 children in Scope
/*  7172*/       OPC_CheckChild0Same, 1,
/*  7174*/       OPC_CheckChild1Same, 0,
/*  7176*/       OPC_MoveParent,
/*  7177*/       OPC_RecordChild1, // #2 = $y
/*  7178*/       OPC_MoveParent,
/*  7179*/       OPC_CheckType, MVT::i32,
/*  7181*/       OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  7183*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7186*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7189*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7192*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7195*/       OPC_EmitInteger, MVT::i32, 1, 
/*  7198*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7201*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7204*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7207*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7210*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7213*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7216*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7228*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7231*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7234*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7237*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7249*/       OPC_EmitInteger, MVT::i32, 1, 
/*  7252*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  7255*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7258*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7261*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 5, 6, 7, 8, 9, 10, 1, 11, 12, 13, 14, 2, 15, 16, 17, 18, 19, 20, 21, 22,  // Results = #23
/*  7287*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7290*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7293*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7305*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7308*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7311*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7323*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7326*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7329*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7341*/       OPC_EmitInteger, MVT::i32, 1, 
/*  7344*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  7347*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7350*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7353*/       OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 3, 4, 23, 24, 25, 26, 0, 27, 28, 29, 2, 30, 31, 32, 33, 34, 35, 36, 
                 // Src: (or:{ *:[i32] } (and:{ *:[i32] } i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$x), (and:{ *:[i32] } (or:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$z), i32:{ *:[i32] }:$y)) - Complexity = 12
                 // Dst: (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$y), i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$y)
/*  7377*/      /*Scope*/ 77|128,1/*205*/, /*->7584*/
/*  7379*/       OPC_CheckChild0Same, 0,
/*  7381*/       OPC_CheckChild1Same, 1,
/*  7383*/       OPC_MoveParent,
/*  7384*/       OPC_RecordChild1, // #2 = $y
/*  7385*/       OPC_MoveParent,
/*  7386*/       OPC_CheckType, MVT::i32,
/*  7388*/       OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  7390*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7393*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7396*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7399*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7402*/       OPC_EmitInteger, MVT::i32, 1, 
/*  7405*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7408*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7411*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7414*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7417*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7420*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7423*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7435*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7438*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7441*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7444*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7456*/       OPC_EmitInteger, MVT::i32, 1, 
/*  7459*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  7462*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7465*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7468*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 5, 6, 7, 8, 9, 10, 1, 11, 12, 13, 14, 2, 15, 16, 17, 18, 19, 20, 21, 22,  // Results = #23
/*  7494*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7497*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7500*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7512*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7515*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7518*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7530*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7533*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7536*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7548*/       OPC_EmitInteger, MVT::i32, 1, 
/*  7551*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  7554*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7557*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7560*/       OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 3, 4, 23, 24, 25, 26, 0, 27, 28, 29, 2, 30, 31, 32, 33, 34, 35, 36, 
                 // Src: (or:{ *:[i32] } (and:{ *:[i32] } i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$x), (and:{ *:[i32] } (or:{ *:[i32] } i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$x), i32:{ *:[i32] }:$y)) - Complexity = 12
                 // Dst: (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$y), i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$y)
/*  7584*/      0, /*End of Scope*/
/*  7585*/     0, /*End of Scope*/
/*  7586*/    /*Scope*/ 109|128,5/*749*/, /*->8337*/
/*  7588*/     OPC_MoveChild1,
/*  7589*/     OPC_CheckOpcode, TARGET_VAL(ISD::OR),
/*  7592*/     OPC_RecordChild0, // #1 = $x
/*  7593*/     OPC_RecordChild1, // #2 = $z
/*  7594*/     OPC_MoveParent,
/*  7595*/     OPC_MoveParent,
/*  7596*/     OPC_MoveChild1,
/*  7597*/     OPC_CheckOpcode, TARGET_VAL(ISD::AND),
/*  7600*/     OPC_Scope, 75|128,1/*203*/, /*->7806*/ // 3 children in Scope
/*  7603*/      OPC_CheckChild0Same, 1,
/*  7605*/      OPC_CheckChild1Same, 2,
/*  7607*/      OPC_MoveParent,
/*  7608*/      OPC_CheckType, MVT::i32,
/*  7610*/      OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  7612*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7615*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7618*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7621*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7624*/      OPC_EmitInteger, MVT::i32, 1, 
/*  7627*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7630*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7633*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7636*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7639*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7642*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7645*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7657*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7660*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7663*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7666*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7678*/      OPC_EmitInteger, MVT::i32, 1, 
/*  7681*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  7684*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7687*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7690*/      OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                    MVT::i32, 20/*#Ops*/, 5, 6, 7, 8, 9, 10, 1, 11, 12, 13, 14, 0, 15, 16, 17, 18, 19, 20, 21, 22,  // Results = #23
/*  7716*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7719*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7722*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7734*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7737*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7740*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7752*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7755*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7758*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7770*/      OPC_EmitInteger, MVT::i32, 1, 
/*  7773*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  7776*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7779*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7782*/      OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 3, 4, 23, 24, 25, 26, 2, 27, 28, 29, 0, 30, 31, 32, 33, 34, 35, 36, 
                // Src: (or:{ *:[i32] } (and:{ *:[i32] } i32:{ *:[i32] }:$y, (or:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$z)), (and:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$z)) - Complexity = 12
                // Dst: (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$y), i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$y)
/*  7806*/     /*Scope*/ 67|128,2/*323*/, /*->8131*/
/*  7808*/      OPC_CheckChild0Same, 2,
/*  7810*/      OPC_CheckChild1Same, 1,
/*  7812*/      OPC_MoveParent,
/*  7813*/      OPC_CheckType, MVT::i32,
/*  7815*/      OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  7817*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7820*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7823*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7826*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7829*/      OPC_EmitInteger, MVT::i32, 1, 
/*  7832*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7835*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7838*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7841*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7844*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7847*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7850*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7862*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7865*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7868*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7871*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7883*/      OPC_EmitInteger, MVT::i32, 1, 
/*  7886*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  7889*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7892*/      OPC_EmitInteger, MVT::i32, 0, 
/*  7895*/      OPC_Scope, 116, /*->8013*/ // 2 children in Scope
/*  7897*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 5, 6, 7, 8, 9, 10, 1, 11, 12, 13, 14, 0, 15, 16, 17, 18, 19, 20, 21, 22,  // Results = #23
/*  7923*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7926*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7929*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7941*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7944*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7947*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7959*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7962*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7965*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  7977*/       OPC_EmitInteger, MVT::i32, 1, 
/*  7980*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  7983*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7986*/       OPC_EmitInteger, MVT::i32, 0, 
/*  7989*/       OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 3, 4, 23, 24, 25, 26, 2, 27, 28, 29, 0, 30, 31, 32, 33, 34, 35, 36, 
                 // Src: (or:{ *:[i32] } (and:{ *:[i32] } i32:{ *:[i32] }:$y, (or:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$z)), (and:{ *:[i32] } i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$x)) - Complexity = 12
                 // Dst: (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$y), i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$y)
/*  8013*/      /*Scope*/ 116, /*->8130*/
/*  8014*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 5, 6, 7, 8, 9, 10, 2, 11, 12, 13, 14, 0, 15, 16, 17, 18, 19, 20, 21, 22,  // Results = #23
/*  8040*/       OPC_EmitInteger, MVT::i32, 0, 
/*  8043*/       OPC_EmitInteger, MVT::i32, 0, 
/*  8046*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8058*/       OPC_EmitInteger, MVT::i32, 0, 
/*  8061*/       OPC_EmitInteger, MVT::i32, 0, 
/*  8064*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8076*/       OPC_EmitInteger, MVT::i32, 0, 
/*  8079*/       OPC_EmitInteger, MVT::i32, 0, 
/*  8082*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8094*/       OPC_EmitInteger, MVT::i32, 1, 
/*  8097*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  8100*/       OPC_EmitInteger, MVT::i32, 0, 
/*  8103*/       OPC_EmitInteger, MVT::i32, 0, 
/*  8106*/       OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 3, 4, 23, 24, 25, 26, 1, 27, 28, 29, 0, 30, 31, 32, 33, 34, 35, 36, 
                 // Src: (or:{ *:[i32] } (and:{ *:[i32] } i32:{ *:[i32] }:$y, (or:{ *:[i32] } i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$x)), (and:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$z)) - Complexity = 12
                 // Dst: (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$y), i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$y)
/*  8130*/      0, /*End of Scope*/
/*  8131*/     /*Scope*/ 75|128,1/*203*/, /*->8336*/
/*  8133*/      OPC_CheckChild0Same, 1,
/*  8135*/      OPC_CheckChild1Same, 2,
/*  8137*/      OPC_MoveParent,
/*  8138*/      OPC_CheckType, MVT::i32,
/*  8140*/      OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  8142*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8145*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8148*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8151*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8154*/      OPC_EmitInteger, MVT::i32, 1, 
/*  8157*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8160*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8163*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8166*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8169*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8172*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8175*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8187*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8190*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8193*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8196*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8208*/      OPC_EmitInteger, MVT::i32, 1, 
/*  8211*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  8214*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8217*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8220*/      OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                    MVT::i32, 20/*#Ops*/, 5, 6, 7, 8, 9, 10, 2, 11, 12, 13, 14, 0, 15, 16, 17, 18, 19, 20, 21, 22,  // Results = #23
/*  8246*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8249*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8252*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8264*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8267*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8270*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8282*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8285*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8288*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8300*/      OPC_EmitInteger, MVT::i32, 1, 
/*  8303*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  8306*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8309*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8312*/      OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 3, 4, 23, 24, 25, 26, 1, 27, 28, 29, 0, 30, 31, 32, 33, 34, 35, 36, 
                // Src: (or:{ *:[i32] } (and:{ *:[i32] } i32:{ *:[i32] }:$y, (or:{ *:[i32] } i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$x)), (and:{ *:[i32] } i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$x)) - Complexity = 12
                // Dst: (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$y), i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$y)
/*  8336*/     0, /*End of Scope*/
/*  8337*/    0, /*End of Scope*/
/*  8338*/   /*Scope*/ 110|128,5/*750*/, /*->9090*/
/*  8340*/    OPC_MoveChild0,
/*  8341*/    OPC_CheckOpcode, TARGET_VAL(ISD::OR),
/*  8344*/    OPC_RecordChild0, // #0 = $x
/*  8345*/    OPC_RecordChild1, // #1 = $z
/*  8346*/    OPC_MoveParent,
/*  8347*/    OPC_RecordChild1, // #2 = $y
/*  8348*/    OPC_MoveParent,
/*  8349*/    OPC_MoveChild1,
/*  8350*/    OPC_CheckOpcode, TARGET_VAL(ISD::AND),
/*  8353*/    OPC_Scope, 75|128,1/*203*/, /*->8559*/ // 3 children in Scope
/*  8356*/     OPC_CheckChild0Same, 0,
/*  8358*/     OPC_CheckChild1Same, 1,
/*  8360*/     OPC_MoveParent,
/*  8361*/     OPC_CheckType, MVT::i32,
/*  8363*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  8365*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8368*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8371*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8374*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8377*/     OPC_EmitInteger, MVT::i32, 1, 
/*  8380*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8383*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8386*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8389*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8392*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8395*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8398*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8410*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8413*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8416*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8419*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8431*/     OPC_EmitInteger, MVT::i32, 1, 
/*  8434*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  8437*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8440*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8443*/     OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                   MVT::i32, 20/*#Ops*/, 5, 6, 7, 8, 9, 10, 0, 11, 12, 13, 14, 2, 15, 16, 17, 18, 19, 20, 21, 22,  // Results = #23
/*  8469*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8472*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8475*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8487*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8490*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8493*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8505*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8508*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8511*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8523*/     OPC_EmitInteger, MVT::i32, 1, 
/*  8526*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  8529*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8532*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8535*/     OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 3, 4, 23, 24, 25, 26, 1, 27, 28, 29, 2, 30, 31, 32, 33, 34, 35, 36, 
               // Src: (or:{ *:[i32] } (and:{ *:[i32] } (or:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$z), i32:{ *:[i32] }:$y), (and:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$z)) - Complexity = 12
               // Dst: (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$y), i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$y)
/*  8559*/    /*Scope*/ 67|128,2/*323*/, /*->8884*/
/*  8561*/     OPC_CheckChild0Same, 1,
/*  8563*/     OPC_CheckChild1Same, 0,
/*  8565*/     OPC_MoveParent,
/*  8566*/     OPC_CheckType, MVT::i32,
/*  8568*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  8570*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8573*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8576*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8579*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8582*/     OPC_EmitInteger, MVT::i32, 1, 
/*  8585*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8588*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8591*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8594*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8597*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8600*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8603*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8615*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8618*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8621*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8624*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8636*/     OPC_EmitInteger, MVT::i32, 1, 
/*  8639*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  8642*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8645*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8648*/     OPC_Scope, 116, /*->8766*/ // 2 children in Scope
/*  8650*/      OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                    MVT::i32, 20/*#Ops*/, 5, 6, 7, 8, 9, 10, 0, 11, 12, 13, 14, 2, 15, 16, 17, 18, 19, 20, 21, 22,  // Results = #23
/*  8676*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8679*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8682*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8694*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8697*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8700*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8712*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8715*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8718*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8730*/      OPC_EmitInteger, MVT::i32, 1, 
/*  8733*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  8736*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8739*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8742*/      OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 3, 4, 23, 24, 25, 26, 1, 27, 28, 29, 2, 30, 31, 32, 33, 34, 35, 36, 
                // Src: (or:{ *:[i32] } (and:{ *:[i32] } (or:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$z), i32:{ *:[i32] }:$y), (and:{ *:[i32] } i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$x)) - Complexity = 12
                // Dst: (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$y), i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$y)
/*  8766*/     /*Scope*/ 116, /*->8883*/
/*  8767*/      OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                    MVT::i32, 20/*#Ops*/, 5, 6, 7, 8, 9, 10, 1, 11, 12, 13, 14, 2, 15, 16, 17, 18, 19, 20, 21, 22,  // Results = #23
/*  8793*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8796*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8799*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8811*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8814*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8817*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8829*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8832*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8835*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8847*/      OPC_EmitInteger, MVT::i32, 1, 
/*  8850*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  8853*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8856*/      OPC_EmitInteger, MVT::i32, 0, 
/*  8859*/      OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 3, 4, 23, 24, 25, 26, 0, 27, 28, 29, 2, 30, 31, 32, 33, 34, 35, 36, 
                // Src: (or:{ *:[i32] } (and:{ *:[i32] } (or:{ *:[i32] } i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$x), i32:{ *:[i32] }:$y), (and:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$z)) - Complexity = 12
                // Dst: (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$y), i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$y)
/*  8883*/     0, /*End of Scope*/
/*  8884*/    /*Scope*/ 75|128,1/*203*/, /*->9089*/
/*  8886*/     OPC_CheckChild0Same, 0,
/*  8888*/     OPC_CheckChild1Same, 1,
/*  8890*/     OPC_MoveParent,
/*  8891*/     OPC_CheckType, MVT::i32,
/*  8893*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  8895*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8898*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8901*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8904*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8907*/     OPC_EmitInteger, MVT::i32, 1, 
/*  8910*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8913*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8916*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8919*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8922*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8925*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8928*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8940*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8943*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8946*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8949*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  8961*/     OPC_EmitInteger, MVT::i32, 1, 
/*  8964*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  8967*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8970*/     OPC_EmitInteger, MVT::i32, 0, 
/*  8973*/     OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                   MVT::i32, 20/*#Ops*/, 5, 6, 7, 8, 9, 10, 1, 11, 12, 13, 14, 2, 15, 16, 17, 18, 19, 20, 21, 22,  // Results = #23
/*  8999*/     OPC_EmitInteger, MVT::i32, 0, 
/*  9002*/     OPC_EmitInteger, MVT::i32, 0, 
/*  9005*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  9017*/     OPC_EmitInteger, MVT::i32, 0, 
/*  9020*/     OPC_EmitInteger, MVT::i32, 0, 
/*  9023*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  9035*/     OPC_EmitInteger, MVT::i32, 0, 
/*  9038*/     OPC_EmitInteger, MVT::i32, 0, 
/*  9041*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  9053*/     OPC_EmitInteger, MVT::i32, 1, 
/*  9056*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  9059*/     OPC_EmitInteger, MVT::i32, 0, 
/*  9062*/     OPC_EmitInteger, MVT::i32, 0, 
/*  9065*/     OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 3, 4, 23, 24, 25, 26, 0, 27, 28, 29, 2, 30, 31, 32, 33, 34, 35, 36, 
               // Src: (or:{ *:[i32] } (and:{ *:[i32] } (or:{ *:[i32] } i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$x), i32:{ *:[i32] }:$y), (and:{ *:[i32] } i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$x)) - Complexity = 12
               // Dst: (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } i32:{ *:[i32] }:$x, i32:{ *:[i32] }:$y), i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$y)
/*  9089*/    0, /*End of Scope*/
/*  9090*/   /*Scope*/ 108|128,47/*6124*/, /*->15216*/
/*  9092*/    OPC_RecordChild0, // #0 = $x
/*  9093*/    OPC_Scope, 14|128,32/*4110*/, /*->13206*/ // 2 children in Scope
/*  9096*/     OPC_RecordChild1, // #1 = $z
/*  9097*/     OPC_MoveParent,
/*  9098*/     OPC_MoveChild1,
/*  9099*/     OPC_CheckOpcode, TARGET_VAL(ISD::AND),
/*  9102*/     OPC_Scope, 127|128,7/*1023*/, /*->10128*/ // 4 children in Scope
/*  9105*/      OPC_RecordChild0, // #2 = $y
/*  9106*/      OPC_MoveChild1,
/*  9107*/      OPC_CheckOpcode, TARGET_VAL(ISD::OR),
/*  9110*/      OPC_Scope, 122|128,3/*506*/, /*->9619*/ // 2 children in Scope
/*  9113*/       OPC_CheckChild0Same, 0,
/*  9115*/       OPC_CheckChild1Same, 1,
/*  9117*/       OPC_MoveParent,
/*  9118*/       OPC_MoveParent,
/*  9119*/       OPC_CheckType, MVT::i64,
/*  9121*/       OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  9123*/       OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/*  9126*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9129*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9132*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9135*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9138*/       OPC_EmitInteger, MVT::i32, 1, 
/*  9141*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9144*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9147*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9150*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/*  9153*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 12,  // Results = #13
/*  9161*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9164*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9167*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9170*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  9182*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/*  9185*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 18,  // Results = #19
/*  9193*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9196*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9199*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9202*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  9214*/       OPC_EmitInteger, MVT::i32, 1, 
/*  9217*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  9220*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9223*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9226*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,  // Results = #28
/*  9252*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9255*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9258*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  9270*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/*  9273*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 1, 32,  // Results = #33
/*  9281*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9284*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9287*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  9299*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/*  9302*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 37,  // Results = #38
/*  9310*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9313*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9316*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  9328*/       OPC_EmitInteger, MVT::i32, 1, 
/*  9331*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  9334*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9337*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9340*/       OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 4, 5, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45,  // Results = #46
/*  9364*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/*  9367*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9370*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9373*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9376*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9379*/       OPC_EmitInteger, MVT::i32, 1, 
/*  9382*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9385*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9388*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9391*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/*  9394*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 56,  // Results = #57
/*  9402*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9405*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9408*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9411*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  9423*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/*  9426*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 62,  // Results = #63
/*  9434*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9437*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9440*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9443*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  9455*/       OPC_EmitInteger, MVT::i32, 1, 
/*  9458*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  9461*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9464*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9467*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71,  // Results = #72
/*  9493*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9496*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9499*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  9511*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/*  9514*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 1, 76,  // Results = #77
/*  9522*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9525*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9528*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  9540*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/*  9543*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 81,  // Results = #82
/*  9551*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9554*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9557*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  9569*/       OPC_EmitInteger, MVT::i32, 1, 
/*  9572*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  9575*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9578*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9581*/       OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 48, 49, 72, 73, 74, 75, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89,  // Results = #90
/*  9605*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/*  9608*/       OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                     MVT::i64, 5/*#Ops*/, 3, 46, 47, 90, 91, 
                 // Src: (or:{ *:[i64] } (and:{ *:[i64] } i64:{ *:[i64] }:$x, i64:{ *:[i64] }:$z), (and:{ *:[i64] } i64:{ *:[i64] }:$y, (or:{ *:[i64] } i64:{ *:[i64] }:$x, i64:{ *:[i64] }:$z))) - Complexity = 12
                 // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/*  9619*/      /*Scope*/ 122|128,3/*506*/, /*->10127*/
/*  9621*/       OPC_CheckChild0Same, 1,
/*  9623*/       OPC_CheckChild1Same, 0,
/*  9625*/       OPC_MoveParent,
/*  9626*/       OPC_MoveParent,
/*  9627*/       OPC_CheckType, MVT::i64,
/*  9629*/       OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/*  9631*/       OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/*  9634*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9637*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9640*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9643*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9646*/       OPC_EmitInteger, MVT::i32, 1, 
/*  9649*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9652*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9655*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9658*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/*  9661*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 12,  // Results = #13
/*  9669*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9672*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9675*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9678*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  9690*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/*  9693*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 18,  // Results = #19
/*  9701*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9704*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9707*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9710*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  9722*/       OPC_EmitInteger, MVT::i32, 1, 
/*  9725*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  9728*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9731*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9734*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,  // Results = #28
/*  9760*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9763*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9766*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  9778*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/*  9781*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 1, 32,  // Results = #33
/*  9789*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9792*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9795*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  9807*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/*  9810*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 37,  // Results = #38
/*  9818*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9821*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9824*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  9836*/       OPC_EmitInteger, MVT::i32, 1, 
/*  9839*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  9842*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9845*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9848*/       OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 4, 5, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45,  // Results = #46
/*  9872*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/*  9875*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9878*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9881*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9884*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9887*/       OPC_EmitInteger, MVT::i32, 1, 
/*  9890*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9893*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9896*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9899*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/*  9902*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 56,  // Results = #57
/*  9910*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9913*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9916*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9919*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  9931*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/*  9934*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 62,  // Results = #63
/*  9942*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9945*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9948*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9951*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*  9963*/       OPC_EmitInteger, MVT::i32, 1, 
/*  9966*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/*  9969*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9972*/       OPC_EmitInteger, MVT::i32, 0, 
/*  9975*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71,  // Results = #72
/* 10001*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10004*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10007*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 10019*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 10022*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 1, 76,  // Results = #77
/* 10030*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10033*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10036*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 10048*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 10051*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 81,  // Results = #82
/* 10059*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10062*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10065*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 10077*/       OPC_EmitInteger, MVT::i32, 1, 
/* 10080*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 10083*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10086*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10089*/       OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 48, 49, 72, 73, 74, 75, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89,  // Results = #90
/* 10113*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 10116*/       OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                     MVT::i64, 5/*#Ops*/, 3, 46, 47, 90, 91, 
                 // Src: (or:{ *:[i64] } (and:{ *:[i64] } i64:{ *:[i64] }:$x, i64:{ *:[i64] }:$z), (and:{ *:[i64] } i64:{ *:[i64] }:$y, (or:{ *:[i64] } i64:{ *:[i64] }:$z, i64:{ *:[i64] }:$x))) - Complexity = 12
                 // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 10127*/      0, /*End of Scope*/
/* 10128*/     /*Scope*/ 0|128,8/*1024*/, /*->11154*/
/* 10130*/      OPC_MoveChild0,
/* 10131*/      OPC_CheckOpcode, TARGET_VAL(ISD::OR),
/* 10134*/      OPC_Scope, 123|128,3/*507*/, /*->10644*/ // 2 children in Scope
/* 10137*/       OPC_CheckChild0Same, 0,
/* 10139*/       OPC_CheckChild1Same, 1,
/* 10141*/       OPC_MoveParent,
/* 10142*/       OPC_RecordChild1, // #2 = $y
/* 10143*/       OPC_MoveParent,
/* 10144*/       OPC_CheckType, MVT::i64,
/* 10146*/       OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 10148*/       OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 10151*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10154*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10157*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10160*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10163*/       OPC_EmitInteger, MVT::i32, 1, 
/* 10166*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10169*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10172*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10175*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 10178*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 12,  // Results = #13
/* 10186*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10189*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10192*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10195*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 10207*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 10210*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 18,  // Results = #19
/* 10218*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10221*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10224*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10227*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 10239*/       OPC_EmitInteger, MVT::i32, 1, 
/* 10242*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 10245*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10248*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10251*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,  // Results = #28
/* 10277*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10280*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10283*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 10295*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 10298*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 1, 32,  // Results = #33
/* 10306*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10309*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10312*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 10324*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 10327*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 37,  // Results = #38
/* 10335*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10338*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10341*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 10353*/       OPC_EmitInteger, MVT::i32, 1, 
/* 10356*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 10359*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10362*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10365*/       OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 4, 5, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45,  // Results = #46
/* 10389*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 10392*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10395*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10398*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10401*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10404*/       OPC_EmitInteger, MVT::i32, 1, 
/* 10407*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10410*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10413*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10416*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 10419*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 56,  // Results = #57
/* 10427*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10430*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10433*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10436*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 10448*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 10451*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 62,  // Results = #63
/* 10459*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10462*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10465*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10468*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 10480*/       OPC_EmitInteger, MVT::i32, 1, 
/* 10483*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 10486*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10489*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10492*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71,  // Results = #72
/* 10518*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10521*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10524*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 10536*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 10539*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 1, 76,  // Results = #77
/* 10547*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10550*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10553*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 10565*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 10568*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 81,  // Results = #82
/* 10576*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10579*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10582*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 10594*/       OPC_EmitInteger, MVT::i32, 1, 
/* 10597*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 10600*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10603*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10606*/       OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 48, 49, 72, 73, 74, 75, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89,  // Results = #90
/* 10630*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 10633*/       OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                     MVT::i64, 5/*#Ops*/, 3, 46, 47, 90, 91, 
                 // Src: (or:{ *:[i64] } (and:{ *:[i64] } i64:{ *:[i64] }:$x, i64:{ *:[i64] }:$z), (and:{ *:[i64] } (or:{ *:[i64] } i64:{ *:[i64] }:$x, i64:{ *:[i64] }:$z), i64:{ *:[i64] }:$y)) - Complexity = 12
                 // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 10644*/      /*Scope*/ 123|128,3/*507*/, /*->11153*/
/* 10646*/       OPC_CheckChild0Same, 1,
/* 10648*/       OPC_CheckChild1Same, 0,
/* 10650*/       OPC_MoveParent,
/* 10651*/       OPC_RecordChild1, // #2 = $y
/* 10652*/       OPC_MoveParent,
/* 10653*/       OPC_CheckType, MVT::i64,
/* 10655*/       OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 10657*/       OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 10660*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10663*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10666*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10669*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10672*/       OPC_EmitInteger, MVT::i32, 1, 
/* 10675*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10678*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10681*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10684*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 10687*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 12,  // Results = #13
/* 10695*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10698*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10701*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10704*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 10716*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 10719*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 18,  // Results = #19
/* 10727*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10730*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10733*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10736*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 10748*/       OPC_EmitInteger, MVT::i32, 1, 
/* 10751*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 10754*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10757*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10760*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,  // Results = #28
/* 10786*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10789*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10792*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 10804*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 10807*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 1, 32,  // Results = #33
/* 10815*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10818*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10821*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 10833*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 10836*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 37,  // Results = #38
/* 10844*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10847*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10850*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 10862*/       OPC_EmitInteger, MVT::i32, 1, 
/* 10865*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 10868*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10871*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10874*/       OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 4, 5, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45,  // Results = #46
/* 10898*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 10901*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10904*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10907*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10910*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10913*/       OPC_EmitInteger, MVT::i32, 1, 
/* 10916*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10919*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10922*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10925*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 10928*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 56,  // Results = #57
/* 10936*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10939*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10942*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10945*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 10957*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 10960*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 62,  // Results = #63
/* 10968*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10971*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10974*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10977*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 10989*/       OPC_EmitInteger, MVT::i32, 1, 
/* 10992*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 10995*/       OPC_EmitInteger, MVT::i32, 0, 
/* 10998*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11001*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71,  // Results = #72
/* 11027*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11030*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11033*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 11045*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 11048*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 1, 76,  // Results = #77
/* 11056*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11059*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11062*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 11074*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 11077*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 81,  // Results = #82
/* 11085*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11088*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11091*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 11103*/       OPC_EmitInteger, MVT::i32, 1, 
/* 11106*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 11109*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11112*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11115*/       OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 48, 49, 72, 73, 74, 75, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89,  // Results = #90
/* 11139*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 11142*/       OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                     MVT::i64, 5/*#Ops*/, 3, 46, 47, 90, 91, 
                 // Src: (or:{ *:[i64] } (and:{ *:[i64] } i64:{ *:[i64] }:$x, i64:{ *:[i64] }:$z), (and:{ *:[i64] } (or:{ *:[i64] } i64:{ *:[i64] }:$z, i64:{ *:[i64] }:$x), i64:{ *:[i64] }:$y)) - Complexity = 12
                 // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 11153*/      0, /*End of Scope*/
/* 11154*/     /*Scope*/ 127|128,7/*1023*/, /*->12179*/
/* 11156*/      OPC_RecordChild0, // #2 = $y
/* 11157*/      OPC_MoveChild1,
/* 11158*/      OPC_CheckOpcode, TARGET_VAL(ISD::OR),
/* 11161*/      OPC_Scope, 122|128,3/*506*/, /*->11670*/ // 2 children in Scope
/* 11164*/       OPC_CheckChild0Same, 1,
/* 11166*/       OPC_CheckChild1Same, 0,
/* 11168*/       OPC_MoveParent,
/* 11169*/       OPC_MoveParent,
/* 11170*/       OPC_CheckType, MVT::i64,
/* 11172*/       OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 11174*/       OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 11177*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11180*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11183*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11186*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11189*/       OPC_EmitInteger, MVT::i32, 1, 
/* 11192*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11195*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11198*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11201*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 11204*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 1, 12,  // Results = #13
/* 11212*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11215*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11218*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11221*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 11233*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 11236*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 18,  // Results = #19
/* 11244*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11247*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11250*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11253*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 11265*/       OPC_EmitInteger, MVT::i32, 1, 
/* 11268*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 11271*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11274*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11277*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,  // Results = #28
/* 11303*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11306*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11309*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 11321*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 11324*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 32,  // Results = #33
/* 11332*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11335*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11338*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 11350*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 11353*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 37,  // Results = #38
/* 11361*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11364*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11367*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 11379*/       OPC_EmitInteger, MVT::i32, 1, 
/* 11382*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 11385*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11388*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11391*/       OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 4, 5, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45,  // Results = #46
/* 11415*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 11418*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11421*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11424*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11427*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11430*/       OPC_EmitInteger, MVT::i32, 1, 
/* 11433*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11436*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11439*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11442*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 11445*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 1, 56,  // Results = #57
/* 11453*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11456*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11459*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11462*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 11474*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 11477*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 62,  // Results = #63
/* 11485*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11488*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11491*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11494*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 11506*/       OPC_EmitInteger, MVT::i32, 1, 
/* 11509*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 11512*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11515*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11518*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71,  // Results = #72
/* 11544*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11547*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11550*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 11562*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 11565*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 76,  // Results = #77
/* 11573*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11576*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11579*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 11591*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 11594*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 81,  // Results = #82
/* 11602*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11605*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11608*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 11620*/       OPC_EmitInteger, MVT::i32, 1, 
/* 11623*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 11626*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11629*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11632*/       OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 48, 49, 72, 73, 74, 75, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89,  // Results = #90
/* 11656*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 11659*/       OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                     MVT::i64, 5/*#Ops*/, 3, 46, 47, 90, 91, 
                 // Src: (or:{ *:[i64] } (and:{ *:[i64] } i64:{ *:[i64] }:$z, i64:{ *:[i64] }:$x), (and:{ *:[i64] } i64:{ *:[i64] }:$y, (or:{ *:[i64] } i64:{ *:[i64] }:$x, i64:{ *:[i64] }:$z))) - Complexity = 12
                 // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 11670*/      /*Scope*/ 122|128,3/*506*/, /*->12178*/
/* 11672*/       OPC_CheckChild0Same, 0,
/* 11674*/       OPC_CheckChild1Same, 1,
/* 11676*/       OPC_MoveParent,
/* 11677*/       OPC_MoveParent,
/* 11678*/       OPC_CheckType, MVT::i64,
/* 11680*/       OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 11682*/       OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 11685*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11688*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11691*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11694*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11697*/       OPC_EmitInteger, MVT::i32, 1, 
/* 11700*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11703*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11706*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11709*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 11712*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 1, 12,  // Results = #13
/* 11720*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11723*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11726*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11729*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 11741*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 11744*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 18,  // Results = #19
/* 11752*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11755*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11758*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11761*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 11773*/       OPC_EmitInteger, MVT::i32, 1, 
/* 11776*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 11779*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11782*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11785*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,  // Results = #28
/* 11811*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11814*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11817*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 11829*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 11832*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 32,  // Results = #33
/* 11840*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11843*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11846*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 11858*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 11861*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 37,  // Results = #38
/* 11869*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11872*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11875*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 11887*/       OPC_EmitInteger, MVT::i32, 1, 
/* 11890*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 11893*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11896*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11899*/       OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 4, 5, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45,  // Results = #46
/* 11923*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 11926*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11929*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11932*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11935*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11938*/       OPC_EmitInteger, MVT::i32, 1, 
/* 11941*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11944*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11947*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11950*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 11953*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 1, 56,  // Results = #57
/* 11961*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11964*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11967*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11970*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 11982*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 11985*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 62,  // Results = #63
/* 11993*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11996*/       OPC_EmitInteger, MVT::i32, 0, 
/* 11999*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12002*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 12014*/       OPC_EmitInteger, MVT::i32, 1, 
/* 12017*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 12020*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12023*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12026*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71,  // Results = #72
/* 12052*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12055*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12058*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 12070*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 12073*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 76,  // Results = #77
/* 12081*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12084*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12087*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 12099*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 12102*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 81,  // Results = #82
/* 12110*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12113*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12116*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 12128*/       OPC_EmitInteger, MVT::i32, 1, 
/* 12131*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 12134*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12137*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12140*/       OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 48, 49, 72, 73, 74, 75, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89,  // Results = #90
/* 12164*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 12167*/       OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                     MVT::i64, 5/*#Ops*/, 3, 46, 47, 90, 91, 
                 // Src: (or:{ *:[i64] } (and:{ *:[i64] } i64:{ *:[i64] }:$z, i64:{ *:[i64] }:$x), (and:{ *:[i64] } i64:{ *:[i64] }:$y, (or:{ *:[i64] } i64:{ *:[i64] }:$z, i64:{ *:[i64] }:$x))) - Complexity = 12
                 // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 12178*/      0, /*End of Scope*/
/* 12179*/     /*Scope*/ 0|128,8/*1024*/, /*->13205*/
/* 12181*/      OPC_MoveChild0,
/* 12182*/      OPC_CheckOpcode, TARGET_VAL(ISD::OR),
/* 12185*/      OPC_Scope, 123|128,3/*507*/, /*->12695*/ // 2 children in Scope
/* 12188*/       OPC_CheckChild0Same, 1,
/* 12190*/       OPC_CheckChild1Same, 0,
/* 12192*/       OPC_MoveParent,
/* 12193*/       OPC_RecordChild1, // #2 = $y
/* 12194*/       OPC_MoveParent,
/* 12195*/       OPC_CheckType, MVT::i64,
/* 12197*/       OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 12199*/       OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 12202*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12205*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12208*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12211*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12214*/       OPC_EmitInteger, MVT::i32, 1, 
/* 12217*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12220*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12223*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12226*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 12229*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 1, 12,  // Results = #13
/* 12237*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12240*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12243*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12246*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 12258*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 12261*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 18,  // Results = #19
/* 12269*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12272*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12275*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12278*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 12290*/       OPC_EmitInteger, MVT::i32, 1, 
/* 12293*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 12296*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12299*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12302*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,  // Results = #28
/* 12328*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12331*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12334*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 12346*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 12349*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 32,  // Results = #33
/* 12357*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12360*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12363*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 12375*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 12378*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 37,  // Results = #38
/* 12386*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12389*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12392*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 12404*/       OPC_EmitInteger, MVT::i32, 1, 
/* 12407*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 12410*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12413*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12416*/       OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 4, 5, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45,  // Results = #46
/* 12440*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 12443*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12446*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12449*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12452*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12455*/       OPC_EmitInteger, MVT::i32, 1, 
/* 12458*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12461*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12464*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12467*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 12470*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 1, 56,  // Results = #57
/* 12478*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12481*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12484*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12487*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 12499*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 12502*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 62,  // Results = #63
/* 12510*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12513*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12516*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12519*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 12531*/       OPC_EmitInteger, MVT::i32, 1, 
/* 12534*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 12537*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12540*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12543*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71,  // Results = #72
/* 12569*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12572*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12575*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 12587*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 12590*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 76,  // Results = #77
/* 12598*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12601*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12604*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 12616*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 12619*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 81,  // Results = #82
/* 12627*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12630*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12633*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 12645*/       OPC_EmitInteger, MVT::i32, 1, 
/* 12648*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 12651*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12654*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12657*/       OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 48, 49, 72, 73, 74, 75, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89,  // Results = #90
/* 12681*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 12684*/       OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                     MVT::i64, 5/*#Ops*/, 3, 46, 47, 90, 91, 
                 // Src: (or:{ *:[i64] } (and:{ *:[i64] } i64:{ *:[i64] }:$z, i64:{ *:[i64] }:$x), (and:{ *:[i64] } (or:{ *:[i64] } i64:{ *:[i64] }:$x, i64:{ *:[i64] }:$z), i64:{ *:[i64] }:$y)) - Complexity = 12
                 // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 12695*/      /*Scope*/ 123|128,3/*507*/, /*->13204*/
/* 12697*/       OPC_CheckChild0Same, 0,
/* 12699*/       OPC_CheckChild1Same, 1,
/* 12701*/       OPC_MoveParent,
/* 12702*/       OPC_RecordChild1, // #2 = $y
/* 12703*/       OPC_MoveParent,
/* 12704*/       OPC_CheckType, MVT::i64,
/* 12706*/       OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 12708*/       OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 12711*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12714*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12717*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12720*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12723*/       OPC_EmitInteger, MVT::i32, 1, 
/* 12726*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12729*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12732*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12735*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 12738*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 1, 12,  // Results = #13
/* 12746*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12749*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12752*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12755*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 12767*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 12770*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 18,  // Results = #19
/* 12778*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12781*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12784*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12787*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 12799*/       OPC_EmitInteger, MVT::i32, 1, 
/* 12802*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 12805*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12808*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12811*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,  // Results = #28
/* 12837*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12840*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12843*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 12855*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 12858*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 32,  // Results = #33
/* 12866*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12869*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12872*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 12884*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 12887*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 37,  // Results = #38
/* 12895*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12898*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12901*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 12913*/       OPC_EmitInteger, MVT::i32, 1, 
/* 12916*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 12919*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12922*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12925*/       OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 4, 5, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45,  // Results = #46
/* 12949*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 12952*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12955*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12958*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12961*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12964*/       OPC_EmitInteger, MVT::i32, 1, 
/* 12967*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12970*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12973*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12976*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 12979*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 1, 56,  // Results = #57
/* 12987*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12990*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12993*/       OPC_EmitInteger, MVT::i32, 0, 
/* 12996*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 13008*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 13011*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 62,  // Results = #63
/* 13019*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13022*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13025*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13028*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 13040*/       OPC_EmitInteger, MVT::i32, 1, 
/* 13043*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 13046*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13049*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13052*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71,  // Results = #72
/* 13078*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13081*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13084*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 13096*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 13099*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 76,  // Results = #77
/* 13107*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13110*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13113*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 13125*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 13128*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 81,  // Results = #82
/* 13136*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13139*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13142*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 13154*/       OPC_EmitInteger, MVT::i32, 1, 
/* 13157*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 13160*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13163*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13166*/       OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 48, 49, 72, 73, 74, 75, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89,  // Results = #90
/* 13190*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 13193*/       OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                     MVT::i64, 5/*#Ops*/, 3, 46, 47, 90, 91, 
                 // Src: (or:{ *:[i64] } (and:{ *:[i64] } i64:{ *:[i64] }:$z, i64:{ *:[i64] }:$x), (and:{ *:[i64] } (or:{ *:[i64] } i64:{ *:[i64] }:$z, i64:{ *:[i64] }:$x), i64:{ *:[i64] }:$y)) - Complexity = 12
                 // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 13204*/      0, /*End of Scope*/
/* 13205*/     0, /*End of Scope*/
/* 13206*/    /*Scope*/ 87|128,15/*2007*/, /*->15215*/
/* 13208*/     OPC_MoveChild1,
/* 13209*/     OPC_CheckOpcode, TARGET_VAL(ISD::OR),
/* 13212*/     OPC_RecordChild0, // #1 = $x
/* 13213*/     OPC_RecordChild1, // #2 = $z
/* 13214*/     OPC_MoveParent,
/* 13215*/     OPC_MoveParent,
/* 13216*/     OPC_MoveChild1,
/* 13217*/     OPC_CheckOpcode, TARGET_VAL(ISD::AND),
/* 13220*/     OPC_Scope, 121|128,3/*505*/, /*->13728*/ // 3 children in Scope
/* 13223*/      OPC_CheckChild0Same, 1,
/* 13225*/      OPC_CheckChild1Same, 2,
/* 13227*/      OPC_MoveParent,
/* 13228*/      OPC_CheckType, MVT::i64,
/* 13230*/      OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 13232*/      OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 13235*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13238*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13241*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13244*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13247*/      OPC_EmitInteger, MVT::i32, 1, 
/* 13250*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13253*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13256*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13259*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/* 13262*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 1, 12,  // Results = #13
/* 13270*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13273*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13276*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13279*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 13291*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/* 13294*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 18,  // Results = #19
/* 13302*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13305*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13308*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13311*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 13323*/      OPC_EmitInteger, MVT::i32, 1, 
/* 13326*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 13329*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13332*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13335*/      OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                    MVT::i32, 20/*#Ops*/, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,  // Results = #28
/* 13361*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13364*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13367*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 13379*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/* 13382*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 32,  // Results = #33
/* 13390*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13393*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13396*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 13408*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/* 13411*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 37,  // Results = #38
/* 13419*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13422*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13425*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 13437*/      OPC_EmitInteger, MVT::i32, 1, 
/* 13440*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 13443*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13446*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13449*/      OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 4, 5, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45,  // Results = #46
/* 13473*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/* 13476*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13479*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13482*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13485*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13488*/      OPC_EmitInteger, MVT::i32, 1, 
/* 13491*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13494*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13497*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13500*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/* 13503*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 1, 56,  // Results = #57
/* 13511*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13514*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13517*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13520*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 13532*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/* 13535*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 62,  // Results = #63
/* 13543*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13546*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13549*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13552*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 13564*/      OPC_EmitInteger, MVT::i32, 1, 
/* 13567*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 13570*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13573*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13576*/      OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                    MVT::i32, 20/*#Ops*/, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71,  // Results = #72
/* 13602*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13605*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13608*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 13620*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/* 13623*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 76,  // Results = #77
/* 13631*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13634*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13637*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 13649*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/* 13652*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 81,  // Results = #82
/* 13660*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13663*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13666*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 13678*/      OPC_EmitInteger, MVT::i32, 1, 
/* 13681*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 13684*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13687*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13690*/      OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 48, 49, 72, 73, 74, 75, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89,  // Results = #90
/* 13714*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/* 13717*/      OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                    MVT::i64, 5/*#Ops*/, 3, 46, 47, 90, 91, 
                // Src: (or:{ *:[i64] } (and:{ *:[i64] } i64:{ *:[i64] }:$y, (or:{ *:[i64] } i64:{ *:[i64] }:$x, i64:{ *:[i64] }:$z)), (and:{ *:[i64] } i64:{ *:[i64] }:$x, i64:{ *:[i64] }:$z)) - Complexity = 12
                // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 13728*/     /*Scope*/ 81|128,7/*977*/, /*->14707*/
/* 13730*/      OPC_CheckChild0Same, 2,
/* 13732*/      OPC_CheckChild1Same, 1,
/* 13734*/      OPC_MoveParent,
/* 13735*/      OPC_CheckType, MVT::i64,
/* 13737*/      OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 13739*/      OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 13742*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13745*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13748*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13751*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13754*/      OPC_EmitInteger, MVT::i32, 1, 
/* 13757*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13760*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13763*/      OPC_EmitInteger, MVT::i32, 0, 
/* 13766*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/* 13769*/      OPC_Scope, 82|128,3/*466*/, /*->14238*/ // 2 children in Scope
/* 13772*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 1, 12,  // Results = #13
/* 13780*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13783*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13786*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13789*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 13801*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 13804*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 18,  // Results = #19
/* 13812*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13815*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13818*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13821*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 13833*/       OPC_EmitInteger, MVT::i32, 1, 
/* 13836*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 13839*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13842*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13845*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,  // Results = #28
/* 13871*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13874*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13877*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 13889*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 13892*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 32,  // Results = #33
/* 13900*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13903*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13906*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 13918*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 13921*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 37,  // Results = #38
/* 13929*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13932*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13935*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 13947*/       OPC_EmitInteger, MVT::i32, 1, 
/* 13950*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 13953*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13956*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13959*/       OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 4, 5, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45,  // Results = #46
/* 13983*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 13986*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13989*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13992*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13995*/       OPC_EmitInteger, MVT::i32, 0, 
/* 13998*/       OPC_EmitInteger, MVT::i32, 1, 
/* 14001*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14004*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14007*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14010*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 14013*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 1, 56,  // Results = #57
/* 14021*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14024*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14027*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14030*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 14042*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 14045*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 62,  // Results = #63
/* 14053*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14056*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14059*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14062*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 14074*/       OPC_EmitInteger, MVT::i32, 1, 
/* 14077*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 14080*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14083*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14086*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71,  // Results = #72
/* 14112*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14115*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14118*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 14130*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 14133*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 76,  // Results = #77
/* 14141*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14144*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14147*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 14159*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 14162*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 81,  // Results = #82
/* 14170*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14173*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14176*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 14188*/       OPC_EmitInteger, MVT::i32, 1, 
/* 14191*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 14194*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14197*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14200*/       OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 48, 49, 72, 73, 74, 75, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89,  // Results = #90
/* 14224*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 14227*/       OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                     MVT::i64, 5/*#Ops*/, 3, 46, 47, 90, 91, 
                 // Src: (or:{ *:[i64] } (and:{ *:[i64] } i64:{ *:[i64] }:$y, (or:{ *:[i64] } i64:{ *:[i64] }:$x, i64:{ *:[i64] }:$z)), (and:{ *:[i64] } i64:{ *:[i64] }:$z, i64:{ *:[i64] }:$x)) - Complexity = 12
                 // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 14238*/      /*Scope*/ 82|128,3/*466*/, /*->14706*/
/* 14240*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 12,  // Results = #13
/* 14248*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14251*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14254*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14257*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 14269*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 14272*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 18,  // Results = #19
/* 14280*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14283*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14286*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14289*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 14301*/       OPC_EmitInteger, MVT::i32, 1, 
/* 14304*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 14307*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14310*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14313*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,  // Results = #28
/* 14339*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14342*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14345*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 14357*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 14360*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 1, 32,  // Results = #33
/* 14368*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14371*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14374*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 14386*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 14389*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 37,  // Results = #38
/* 14397*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14400*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14403*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 14415*/       OPC_EmitInteger, MVT::i32, 1, 
/* 14418*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 14421*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14424*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14427*/       OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 4, 5, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45,  // Results = #46
/* 14451*/       OPC_EmitInteger, MVT::i32, R600::sub0,
/* 14454*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14457*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14460*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14463*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14466*/       OPC_EmitInteger, MVT::i32, 1, 
/* 14469*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14472*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14475*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14478*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 14481*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 2, 56,  // Results = #57
/* 14489*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14492*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14495*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14498*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 14510*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 14513*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 62,  // Results = #63
/* 14521*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14524*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14527*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14530*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 14542*/       OPC_EmitInteger, MVT::i32, 1, 
/* 14545*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 14548*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14551*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14554*/       OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                     MVT::i32, 20/*#Ops*/, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71,  // Results = #72
/* 14580*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14583*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14586*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 14598*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 14601*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 1, 76,  // Results = #77
/* 14609*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14612*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14615*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 14627*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 14630*/       OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                     MVT::i32, 2/*#Ops*/, 0, 81,  // Results = #82
/* 14638*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14641*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14644*/       OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 14656*/       OPC_EmitInteger, MVT::i32, 1, 
/* 14659*/       OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 14662*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14665*/       OPC_EmitInteger, MVT::i32, 0, 
/* 14668*/       OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                     MVT::i32, 18/*#Ops*/, 48, 49, 72, 73, 74, 75, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89,  // Results = #90
/* 14692*/       OPC_EmitInteger, MVT::i32, R600::sub1,
/* 14695*/       OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                     MVT::i64, 5/*#Ops*/, 3, 46, 47, 90, 91, 
                 // Src: (or:{ *:[i64] } (and:{ *:[i64] } i64:{ *:[i64] }:$y, (or:{ *:[i64] } i64:{ *:[i64] }:$z, i64:{ *:[i64] }:$x)), (and:{ *:[i64] } i64:{ *:[i64] }:$x, i64:{ *:[i64] }:$z)) - Complexity = 12
                 // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 14706*/      0, /*End of Scope*/
/* 14707*/     /*Scope*/ 121|128,3/*505*/, /*->15214*/
/* 14709*/      OPC_CheckChild0Same, 1,
/* 14711*/      OPC_CheckChild1Same, 2,
/* 14713*/      OPC_MoveParent,
/* 14714*/      OPC_CheckType, MVT::i64,
/* 14716*/      OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 14718*/      OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 14721*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14724*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14727*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14730*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14733*/      OPC_EmitInteger, MVT::i32, 1, 
/* 14736*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14739*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14742*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14745*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/* 14748*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 12,  // Results = #13
/* 14756*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14759*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14762*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14765*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 14777*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/* 14780*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 18,  // Results = #19
/* 14788*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14791*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14794*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14797*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 14809*/      OPC_EmitInteger, MVT::i32, 1, 
/* 14812*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 14815*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14818*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14821*/      OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                    MVT::i32, 20/*#Ops*/, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,  // Results = #28
/* 14847*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14850*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14853*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 14865*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/* 14868*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 1, 32,  // Results = #33
/* 14876*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14879*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14882*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 14894*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/* 14897*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 37,  // Results = #38
/* 14905*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14908*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14911*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 14923*/      OPC_EmitInteger, MVT::i32, 1, 
/* 14926*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 14929*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14932*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14935*/      OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 4, 5, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45,  // Results = #46
/* 14959*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/* 14962*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14965*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14968*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14971*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14974*/      OPC_EmitInteger, MVT::i32, 1, 
/* 14977*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14980*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14983*/      OPC_EmitInteger, MVT::i32, 0, 
/* 14986*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/* 14989*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 56,  // Results = #57
/* 14997*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15000*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15003*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15006*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 15018*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/* 15021*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 62,  // Results = #63
/* 15029*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15032*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15035*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15038*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 15050*/      OPC_EmitInteger, MVT::i32, 1, 
/* 15053*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 15056*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15059*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15062*/      OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                    MVT::i32, 20/*#Ops*/, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71,  // Results = #72
/* 15088*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15091*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15094*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 15106*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/* 15109*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 1, 76,  // Results = #77
/* 15117*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15120*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15123*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 15135*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/* 15138*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 81,  // Results = #82
/* 15146*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15149*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15152*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 15164*/      OPC_EmitInteger, MVT::i32, 1, 
/* 15167*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 15170*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15173*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15176*/      OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 48, 49, 72, 73, 74, 75, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89,  // Results = #90
/* 15200*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/* 15203*/      OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                    MVT::i64, 5/*#Ops*/, 3, 46, 47, 90, 91, 
                // Src: (or:{ *:[i64] } (and:{ *:[i64] } i64:{ *:[i64] }:$y, (or:{ *:[i64] } i64:{ *:[i64] }:$z, i64:{ *:[i64] }:$x)), (and:{ *:[i64] } i64:{ *:[i64] }:$z, i64:{ *:[i64] }:$x)) - Complexity = 12
                // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 15214*/     0, /*End of Scope*/
/* 15215*/    0, /*End of Scope*/
/* 15216*/   /*Scope*/ 88|128,15/*2008*/, /*->17226*/
/* 15218*/    OPC_MoveChild0,
/* 15219*/    OPC_CheckOpcode, TARGET_VAL(ISD::OR),
/* 15222*/    OPC_RecordChild0, // #0 = $x
/* 15223*/    OPC_RecordChild1, // #1 = $z
/* 15224*/    OPC_MoveParent,
/* 15225*/    OPC_RecordChild1, // #2 = $y
/* 15226*/    OPC_MoveParent,
/* 15227*/    OPC_MoveChild1,
/* 15228*/    OPC_CheckOpcode, TARGET_VAL(ISD::AND),
/* 15231*/    OPC_Scope, 121|128,3/*505*/, /*->15739*/ // 3 children in Scope
/* 15234*/     OPC_CheckChild0Same, 0,
/* 15236*/     OPC_CheckChild1Same, 1,
/* 15238*/     OPC_MoveParent,
/* 15239*/     OPC_CheckType, MVT::i64,
/* 15241*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 15243*/     OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 15246*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15249*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15252*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15255*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15258*/     OPC_EmitInteger, MVT::i32, 1, 
/* 15261*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15264*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15267*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15270*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 15273*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 12,  // Results = #13
/* 15281*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15284*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15287*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15290*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 15302*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 15305*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 18,  // Results = #19
/* 15313*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15316*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15319*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15322*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 15334*/     OPC_EmitInteger, MVT::i32, 1, 
/* 15337*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 15340*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15343*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15346*/     OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                   MVT::i32, 20/*#Ops*/, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,  // Results = #28
/* 15372*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15375*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15378*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 15390*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 15393*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 32,  // Results = #33
/* 15401*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15404*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15407*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 15419*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 15422*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 37,  // Results = #38
/* 15430*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15433*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15436*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 15448*/     OPC_EmitInteger, MVT::i32, 1, 
/* 15451*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 15454*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15457*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15460*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 4, 5, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45,  // Results = #46
/* 15484*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 15487*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15490*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15493*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15496*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15499*/     OPC_EmitInteger, MVT::i32, 1, 
/* 15502*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15505*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15508*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15511*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 15514*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 56,  // Results = #57
/* 15522*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15525*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15528*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15531*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 15543*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 15546*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 62,  // Results = #63
/* 15554*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15557*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15560*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15563*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 15575*/     OPC_EmitInteger, MVT::i32, 1, 
/* 15578*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 15581*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15584*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15587*/     OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                   MVT::i32, 20/*#Ops*/, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71,  // Results = #72
/* 15613*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15616*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15619*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 15631*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 15634*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 76,  // Results = #77
/* 15642*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15645*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15648*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 15660*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 15663*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 81,  // Results = #82
/* 15671*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15674*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15677*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 15689*/     OPC_EmitInteger, MVT::i32, 1, 
/* 15692*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 15695*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15698*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15701*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 48, 49, 72, 73, 74, 75, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89,  // Results = #90
/* 15725*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 15728*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                   MVT::i64, 5/*#Ops*/, 3, 46, 47, 90, 91, 
               // Src: (or:{ *:[i64] } (and:{ *:[i64] } (or:{ *:[i64] } i64:{ *:[i64] }:$x, i64:{ *:[i64] }:$z), i64:{ *:[i64] }:$y), (and:{ *:[i64] } i64:{ *:[i64] }:$x, i64:{ *:[i64] }:$z)) - Complexity = 12
               // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 15739*/    /*Scope*/ 81|128,7/*977*/, /*->16718*/
/* 15741*/     OPC_CheckChild0Same, 1,
/* 15743*/     OPC_CheckChild1Same, 0,
/* 15745*/     OPC_MoveParent,
/* 15746*/     OPC_CheckType, MVT::i64,
/* 15748*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 15750*/     OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 15753*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15756*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15759*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15762*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15765*/     OPC_EmitInteger, MVT::i32, 1, 
/* 15768*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15771*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15774*/     OPC_EmitInteger, MVT::i32, 0, 
/* 15777*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 15780*/     OPC_Scope, 82|128,3/*466*/, /*->16249*/ // 2 children in Scope
/* 15783*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 12,  // Results = #13
/* 15791*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15794*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15797*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15800*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 15812*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/* 15815*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 18,  // Results = #19
/* 15823*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15826*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15829*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15832*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 15844*/      OPC_EmitInteger, MVT::i32, 1, 
/* 15847*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 15850*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15853*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15856*/      OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                    MVT::i32, 20/*#Ops*/, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,  // Results = #28
/* 15882*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15885*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15888*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 15900*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/* 15903*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 1, 32,  // Results = #33
/* 15911*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15914*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15917*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 15929*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/* 15932*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 37,  // Results = #38
/* 15940*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15943*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15946*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 15958*/      OPC_EmitInteger, MVT::i32, 1, 
/* 15961*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 15964*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15967*/      OPC_EmitInteger, MVT::i32, 0, 
/* 15970*/      OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 4, 5, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45,  // Results = #46
/* 15994*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/* 15997*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16000*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16003*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16006*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16009*/      OPC_EmitInteger, MVT::i32, 1, 
/* 16012*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16015*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16018*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16021*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/* 16024*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 56,  // Results = #57
/* 16032*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16035*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16038*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16041*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 16053*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/* 16056*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 62,  // Results = #63
/* 16064*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16067*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16070*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16073*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 16085*/      OPC_EmitInteger, MVT::i32, 1, 
/* 16088*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 16091*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16094*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16097*/      OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                    MVT::i32, 20/*#Ops*/, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71,  // Results = #72
/* 16123*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16126*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16129*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 16141*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/* 16144*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 1, 76,  // Results = #77
/* 16152*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16155*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16158*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 16170*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/* 16173*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 81,  // Results = #82
/* 16181*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16184*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16187*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 16199*/      OPC_EmitInteger, MVT::i32, 1, 
/* 16202*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 16205*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16208*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16211*/      OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 48, 49, 72, 73, 74, 75, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89,  // Results = #90
/* 16235*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/* 16238*/      OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                    MVT::i64, 5/*#Ops*/, 3, 46, 47, 90, 91, 
                // Src: (or:{ *:[i64] } (and:{ *:[i64] } (or:{ *:[i64] } i64:{ *:[i64] }:$x, i64:{ *:[i64] }:$z), i64:{ *:[i64] }:$y), (and:{ *:[i64] } i64:{ *:[i64] }:$z, i64:{ *:[i64] }:$x)) - Complexity = 12
                // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 16249*/     /*Scope*/ 82|128,3/*466*/, /*->16717*/
/* 16251*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 1, 12,  // Results = #13
/* 16259*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16262*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16265*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16268*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 16280*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/* 16283*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 18,  // Results = #19
/* 16291*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16294*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16297*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16300*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 16312*/      OPC_EmitInteger, MVT::i32, 1, 
/* 16315*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 16318*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16321*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16324*/      OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                    MVT::i32, 20/*#Ops*/, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,  // Results = #28
/* 16350*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16353*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16356*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 16368*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/* 16371*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 32,  // Results = #33
/* 16379*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16382*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16385*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 16397*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/* 16400*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 37,  // Results = #38
/* 16408*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16411*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16414*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 16426*/      OPC_EmitInteger, MVT::i32, 1, 
/* 16429*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 16432*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16435*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16438*/      OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 4, 5, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45,  // Results = #46
/* 16462*/      OPC_EmitInteger, MVT::i32, R600::sub0,
/* 16465*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16468*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16471*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16474*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16477*/      OPC_EmitInteger, MVT::i32, 1, 
/* 16480*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16483*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16486*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16489*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/* 16492*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 1, 56,  // Results = #57
/* 16500*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16503*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16506*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16509*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 16521*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/* 16524*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 62,  // Results = #63
/* 16532*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16535*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16538*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16541*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 16553*/      OPC_EmitInteger, MVT::i32, 1, 
/* 16556*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 16559*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16562*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16565*/      OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                    MVT::i32, 20/*#Ops*/, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71,  // Results = #72
/* 16591*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16594*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16597*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 16609*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/* 16612*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 0, 76,  // Results = #77
/* 16620*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16623*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16626*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 16638*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/* 16641*/      OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                    MVT::i32, 2/*#Ops*/, 2, 81,  // Results = #82
/* 16649*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16652*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16655*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 16667*/      OPC_EmitInteger, MVT::i32, 1, 
/* 16670*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 16673*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16676*/      OPC_EmitInteger, MVT::i32, 0, 
/* 16679*/      OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                    MVT::i32, 18/*#Ops*/, 48, 49, 72, 73, 74, 75, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89,  // Results = #90
/* 16703*/      OPC_EmitInteger, MVT::i32, R600::sub1,
/* 16706*/      OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                    MVT::i64, 5/*#Ops*/, 3, 46, 47, 90, 91, 
                // Src: (or:{ *:[i64] } (and:{ *:[i64] } (or:{ *:[i64] } i64:{ *:[i64] }:$z, i64:{ *:[i64] }:$x), i64:{ *:[i64] }:$y), (and:{ *:[i64] } i64:{ *:[i64] }:$x, i64:{ *:[i64] }:$z)) - Complexity = 12
                // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 16717*/     0, /*End of Scope*/
/* 16718*/    /*Scope*/ 121|128,3/*505*/, /*->17225*/
/* 16720*/     OPC_CheckChild0Same, 0,
/* 16722*/     OPC_CheckChild1Same, 1,
/* 16724*/     OPC_MoveParent,
/* 16725*/     OPC_CheckType, MVT::i64,
/* 16727*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 16729*/     OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 16732*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16735*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16738*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16741*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16744*/     OPC_EmitInteger, MVT::i32, 1, 
/* 16747*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16750*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16753*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16756*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 16759*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 12,  // Results = #13
/* 16767*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16770*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16773*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16776*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 16788*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 16791*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 18,  // Results = #19
/* 16799*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16802*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16805*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16808*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 16820*/     OPC_EmitInteger, MVT::i32, 1, 
/* 16823*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 16826*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16829*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16832*/     OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                   MVT::i32, 20/*#Ops*/, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,  // Results = #28
/* 16858*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16861*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16864*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 16876*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 16879*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 32,  // Results = #33
/* 16887*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16890*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16893*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 16905*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 16908*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 37,  // Results = #38
/* 16916*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16919*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16922*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 16934*/     OPC_EmitInteger, MVT::i32, 1, 
/* 16937*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 16940*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16943*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16946*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 4, 5, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45,  // Results = #46
/* 16970*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 16973*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16976*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16979*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16982*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16985*/     OPC_EmitInteger, MVT::i32, 1, 
/* 16988*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16991*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16994*/     OPC_EmitInteger, MVT::i32, 0, 
/* 16997*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 17000*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 56,  // Results = #57
/* 17008*/     OPC_EmitInteger, MVT::i32, 0, 
/* 17011*/     OPC_EmitInteger, MVT::i32, 0, 
/* 17014*/     OPC_EmitInteger, MVT::i32, 0, 
/* 17017*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 17029*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 17032*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 62,  // Results = #63
/* 17040*/     OPC_EmitInteger, MVT::i32, 0, 
/* 17043*/     OPC_EmitInteger, MVT::i32, 0, 
/* 17046*/     OPC_EmitInteger, MVT::i32, 0, 
/* 17049*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 17061*/     OPC_EmitInteger, MVT::i32, 1, 
/* 17064*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 17067*/     OPC_EmitInteger, MVT::i32, 0, 
/* 17070*/     OPC_EmitInteger, MVT::i32, 0, 
/* 17073*/     OPC_EmitNode1, TARGET_VAL(R600::XOR_INT), 0,
                   MVT::i32, 20/*#Ops*/, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71,  // Results = #72
/* 17099*/     OPC_EmitInteger, MVT::i32, 0, 
/* 17102*/     OPC_EmitInteger, MVT::i32, 0, 
/* 17105*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 17117*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 17120*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 76,  // Results = #77
/* 17128*/     OPC_EmitInteger, MVT::i32, 0, 
/* 17131*/     OPC_EmitInteger, MVT::i32, 0, 
/* 17134*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 17146*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 17149*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 81,  // Results = #82
/* 17157*/     OPC_EmitInteger, MVT::i32, 0, 
/* 17160*/     OPC_EmitInteger, MVT::i32, 0, 
/* 17163*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 17175*/     OPC_EmitInteger, MVT::i32, 1, 
/* 17178*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 17181*/     OPC_EmitInteger, MVT::i32, 0, 
/* 17184*/     OPC_EmitInteger, MVT::i32, 0, 
/* 17187*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 48, 49, 72, 73, 74, 75, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89,  // Results = #90
/* 17211*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 17214*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                   MVT::i64, 5/*#Ops*/, 3, 46, 47, 90, 91, 
               // Src: (or:{ *:[i64] } (and:{ *:[i64] } (or:{ *:[i64] } i64:{ *:[i64] }:$z, i64:{ *:[i64] }:$x), i64:{ *:[i64] }:$y), (and:{ *:[i64] } i64:{ *:[i64] }:$z, i64:{ *:[i64] }:$x)) - Complexity = 12
               // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (XOR_INT:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 17225*/    0, /*End of Scope*/
/* 17226*/   0, /*End of Scope*/
/* 17227*/  /*Scope*/ 104, /*->17332*/
/* 17228*/   OPC_RecordChild0, // #0 = $src0
/* 17229*/   OPC_RecordChild1, // #1 = $src1
/* 17230*/   OPC_CheckType, MVT::i32,
/* 17232*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 17234*/   OPC_EmitInteger, MVT::i32, 0, 
/* 17237*/   OPC_EmitInteger, MVT::i32, 0, 
/* 17240*/   OPC_EmitInteger, MVT::i32, 1, 
/* 17243*/   OPC_EmitInteger, MVT::i32, 0, 
/* 17246*/   OPC_EmitInteger, MVT::i32, 0, 
/* 17249*/   OPC_EmitInteger, MVT::i32, 0, 
/* 17252*/   OPC_EmitInteger, MVT::i32, 0, 
/* 17255*/   OPC_EmitInteger, MVT::i32, 0, 
/* 17258*/   OPC_EmitInteger, MVT::i32, 0, 
/* 17261*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 17273*/   OPC_EmitInteger, MVT::i32, 0, 
/* 17276*/   OPC_EmitInteger, MVT::i32, 0, 
/* 17279*/   OPC_EmitInteger, MVT::i32, 0, 
/* 17282*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 17294*/   OPC_EmitInteger, MVT::i32, 1, 
/* 17297*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 17300*/   OPC_EmitInteger, MVT::i32, 0, 
/* 17303*/   OPC_EmitInteger, MVT::i32, 0, 
/* 17306*/   OPC_MorphNodeTo1, TARGET_VAL(R600::OR_INT), 0,
                 MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
             // Src: (or:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
             // Dst: (OR_INT:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 17332*/  0, /*End of Scope*/
/* 17333*/ /*SwitchOpcode*/ 25, TARGET_VAL(AMDGPUISD::REGISTER_LOAD),// ->17361
/* 17336*/  OPC_RecordNode, // #0 = 'AMDGPUregister_load' chained node
/* 17337*/  OPC_RecordChild1, // #1 = $addr
/* 17338*/  OPC_RecordChild2, // #2 = $chan
/* 17339*/  OPC_MoveChild2,
/* 17340*/  OPC_CheckOpcode, TARGET_VAL(ISD::TargetConstant),
/* 17343*/  OPC_CheckType, MVT::i32,
/* 17345*/  OPC_MoveParent,
/* 17346*/  OPC_CheckType, MVT::i32,
/* 17348*/  OPC_CheckComplexPat, /*CP*/0, /*#*/1, // SelectADDRIndirect:$addr #3 #4
/* 17351*/  OPC_EmitMergeInputChains1_0,
/* 17352*/  OPC_MorphNodeTo1, TARGET_VAL(R600::R600_RegisterLoad), 0|OPFL_Chain,
                MVT::i32, 3/*#Ops*/, 3, 4, 2, 
            // Src: (AMDGPUregister_load:{ *:[i32] } ADDRIndirect:{ *:[iPTR] }:$addr, (timm:{ *:[i32] }):$chan) - Complexity = 15
            // Dst: (R600_RegisterLoad:{ *:[i32] } ADDRIndirect:{ *:[iPTR] }:$addr, (timm:{ *:[i32] }):$chan)
/* 17361*/ /*SwitchOpcode*/ 26, TARGET_VAL(AMDGPUISD::REGISTER_STORE),// ->17390
/* 17364*/  OPC_RecordNode, // #0 = 'AMDGPUregister_store' chained node
/* 17365*/  OPC_RecordChild1, // #1 = $val
/* 17366*/  OPC_CheckChild1Type, MVT::i32,
/* 17368*/  OPC_RecordChild2, // #2 = $addr
/* 17369*/  OPC_RecordChild3, // #3 = $chan
/* 17370*/  OPC_MoveChild3,
/* 17371*/  OPC_CheckOpcode, TARGET_VAL(ISD::TargetConstant),
/* 17374*/  OPC_CheckType, MVT::i32,
/* 17376*/  OPC_MoveParent,
/* 17377*/  OPC_CheckComplexPat, /*CP*/0, /*#*/2, // SelectADDRIndirect:$addr #4 #5
/* 17380*/  OPC_EmitMergeInputChains1_0,
/* 17381*/  OPC_MorphNodeTo0, TARGET_VAL(R600::R600_RegisterStore), 0|OPFL_Chain,
                4/*#Ops*/, 1, 4, 5, 3, 
            // Src: (AMDGPUregister_store i32:{ *:[i32] }:$val, ADDRIndirect:{ *:[iPTR] }:$addr, (timm:{ *:[i32] }):$chan) - Complexity = 15
            // Dst: (R600_RegisterStore i32:{ *:[i32] }:$val, ADDRIndirect:{ *:[iPTR] }:$addr, (timm:{ *:[i32] }):$chan)
/* 17390*/ /*SwitchOpcode*/ 45|128,35/*4525*/, TARGET_VAL(ISD::SELECT_CC),// ->21919
/* 17394*/  OPC_RecordChild0, // #0 = $src0
/* 17395*/  OPC_Scope, 60|128,23/*3004*/, /*->20402*/ // 2 children in Scope
/* 17398*/   OPC_CheckChild0Type, MVT::f32,
/* 17400*/   OPC_Scope, 72|128,13/*1736*/, /*->19139*/ // 2 children in Scope
/* 17403*/    OPC_RecordChild1, // #1 = $src1
/* 17404*/    OPC_Scope, 96|128,6/*864*/, /*->18271*/ // 2 children in Scope
/* 17407*/     OPC_CheckChild2Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 17418*/     OPC_CheckChild3Integer, 0, 
/* 17420*/     OPC_MoveChild4,
/* 17421*/     OPC_Scope, 105, /*->17528*/ // 8 children in Scope
/* 17423*/      OPC_CheckCondCode, ISD::SETOEQ,
/* 17425*/      OPC_MoveParent,
/* 17426*/      OPC_CheckType, MVT::i32,
/* 17428*/      OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 17430*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17433*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17436*/      OPC_EmitInteger, MVT::i32, 1, 
/* 17439*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17442*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17445*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17448*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17451*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17454*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17457*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 17469*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17472*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17475*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17478*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 17490*/      OPC_EmitInteger, MVT::i32, 1, 
/* 17493*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 17496*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17499*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17502*/      OPC_MorphNodeTo1, TARGET_VAL(R600::SETE_DX10), 0,
                    MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
                // Src: (selectcc:{ *:[i32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, -1:{ *:[i32] }, 0:{ *:[i32] }, SETOEQ:{ *:[Other] }) - Complexity = 13
                // Dst: (SETE_DX10:{ *:[i32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1)
/* 17528*/     /*Scope*/ 105, /*->17634*/
/* 17529*/      OPC_CheckCondCode, ISD::SETEQ,
/* 17531*/      OPC_MoveParent,
/* 17532*/      OPC_CheckType, MVT::i32,
/* 17534*/      OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 17536*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17539*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17542*/      OPC_EmitInteger, MVT::i32, 1, 
/* 17545*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17548*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17551*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17554*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17557*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17560*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17563*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 17575*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17578*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17581*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17584*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 17596*/      OPC_EmitInteger, MVT::i32, 1, 
/* 17599*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 17602*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17605*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17608*/      OPC_MorphNodeTo1, TARGET_VAL(R600::SETE_DX10), 0,
                    MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
                // Src: (selectcc:{ *:[i32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, -1:{ *:[i32] }, 0:{ *:[i32] }, SETEQ:{ *:[Other] }) - Complexity = 13
                // Dst: (SETE_DX10:{ *:[i32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1)
/* 17634*/     /*Scope*/ 105, /*->17740*/
/* 17635*/      OPC_CheckCondCode, ISD::SETOGT,
/* 17637*/      OPC_MoveParent,
/* 17638*/      OPC_CheckType, MVT::i32,
/* 17640*/      OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 17642*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17645*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17648*/      OPC_EmitInteger, MVT::i32, 1, 
/* 17651*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17654*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17657*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17660*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17663*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17666*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17669*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 17681*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17684*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17687*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17690*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 17702*/      OPC_EmitInteger, MVT::i32, 1, 
/* 17705*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 17708*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17711*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17714*/      OPC_MorphNodeTo1, TARGET_VAL(R600::SETGT_DX10), 0,
                    MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
                // Src: (selectcc:{ *:[i32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, -1:{ *:[i32] }, 0:{ *:[i32] }, SETOGT:{ *:[Other] }) - Complexity = 13
                // Dst: (SETGT_DX10:{ *:[i32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1)
/* 17740*/     /*Scope*/ 105, /*->17846*/
/* 17741*/      OPC_CheckCondCode, ISD::SETGT,
/* 17743*/      OPC_MoveParent,
/* 17744*/      OPC_CheckType, MVT::i32,
/* 17746*/      OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 17748*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17751*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17754*/      OPC_EmitInteger, MVT::i32, 1, 
/* 17757*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17760*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17763*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17766*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17769*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17772*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17775*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 17787*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17790*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17793*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17796*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 17808*/      OPC_EmitInteger, MVT::i32, 1, 
/* 17811*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 17814*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17817*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17820*/      OPC_MorphNodeTo1, TARGET_VAL(R600::SETGT_DX10), 0,
                    MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
                // Src: (selectcc:{ *:[i32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, -1:{ *:[i32] }, 0:{ *:[i32] }, SETGT:{ *:[Other] }) - Complexity = 13
                // Dst: (SETGT_DX10:{ *:[i32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1)
/* 17846*/     /*Scope*/ 105, /*->17952*/
/* 17847*/      OPC_CheckCondCode, ISD::SETOGE,
/* 17849*/      OPC_MoveParent,
/* 17850*/      OPC_CheckType, MVT::i32,
/* 17852*/      OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 17854*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17857*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17860*/      OPC_EmitInteger, MVT::i32, 1, 
/* 17863*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17866*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17869*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17872*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17875*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17878*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17881*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 17893*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17896*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17899*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17902*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 17914*/      OPC_EmitInteger, MVT::i32, 1, 
/* 17917*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 17920*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17923*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17926*/      OPC_MorphNodeTo1, TARGET_VAL(R600::SETGE_DX10), 0,
                    MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
                // Src: (selectcc:{ *:[i32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, -1:{ *:[i32] }, 0:{ *:[i32] }, SETOGE:{ *:[Other] }) - Complexity = 13
                // Dst: (SETGE_DX10:{ *:[i32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1)
/* 17952*/     /*Scope*/ 105, /*->18058*/
/* 17953*/      OPC_CheckCondCode, ISD::SETGE,
/* 17955*/      OPC_MoveParent,
/* 17956*/      OPC_CheckType, MVT::i32,
/* 17958*/      OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 17960*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17963*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17966*/      OPC_EmitInteger, MVT::i32, 1, 
/* 17969*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17972*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17975*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17978*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17981*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17984*/      OPC_EmitInteger, MVT::i32, 0, 
/* 17987*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 17999*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18002*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18005*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18008*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 18020*/      OPC_EmitInteger, MVT::i32, 1, 
/* 18023*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 18026*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18029*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18032*/      OPC_MorphNodeTo1, TARGET_VAL(R600::SETGE_DX10), 0,
                    MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
                // Src: (selectcc:{ *:[i32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, -1:{ *:[i32] }, 0:{ *:[i32] }, SETGE:{ *:[Other] }) - Complexity = 13
                // Dst: (SETGE_DX10:{ *:[i32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1)
/* 18058*/     /*Scope*/ 105, /*->18164*/
/* 18059*/      OPC_CheckCondCode, ISD::SETUNE,
/* 18061*/      OPC_MoveParent,
/* 18062*/      OPC_CheckType, MVT::i32,
/* 18064*/      OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 18066*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18069*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18072*/      OPC_EmitInteger, MVT::i32, 1, 
/* 18075*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18078*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18081*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18084*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18087*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18090*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18093*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 18105*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18108*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18111*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18114*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 18126*/      OPC_EmitInteger, MVT::i32, 1, 
/* 18129*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 18132*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18135*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18138*/      OPC_MorphNodeTo1, TARGET_VAL(R600::SETNE_DX10), 0,
                    MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
                // Src: (selectcc:{ *:[i32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, -1:{ *:[i32] }, 0:{ *:[i32] }, SETUNE:{ *:[Other] }) - Complexity = 13
                // Dst: (SETNE_DX10:{ *:[i32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1)
/* 18164*/     /*Scope*/ 105, /*->18270*/
/* 18165*/      OPC_CheckCondCode, ISD::SETNE,
/* 18167*/      OPC_MoveParent,
/* 18168*/      OPC_CheckType, MVT::i32,
/* 18170*/      OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 18172*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18175*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18178*/      OPC_EmitInteger, MVT::i32, 1, 
/* 18181*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18184*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18187*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18190*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18193*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18196*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18199*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 18211*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18214*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18217*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18220*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 18232*/      OPC_EmitInteger, MVT::i32, 1, 
/* 18235*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 18238*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18241*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18244*/      OPC_MorphNodeTo1, TARGET_VAL(R600::SETNE_DX10), 0,
                    MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
                // Src: (selectcc:{ *:[i32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, -1:{ *:[i32] }, 0:{ *:[i32] }, SETNE:{ *:[Other] }) - Complexity = 13
                // Dst: (SETNE_DX10:{ *:[i32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1)
/* 18270*/     0, /*End of Scope*/
/* 18271*/    /*Scope*/ 97|128,6/*865*/, /*->19138*/
/* 18273*/     OPC_MoveChild2,
/* 18274*/     OPC_CheckOpcode, TARGET_VAL(ISD::ConstantFP),
/* 18277*/     OPC_CheckPredicate, 2, // Predicate_FP_ONE
/* 18279*/     OPC_MoveParent,
/* 18280*/     OPC_MoveChild3,
/* 18281*/     OPC_CheckOpcode, TARGET_VAL(ISD::ConstantFP),
/* 18284*/     OPC_CheckPredicate, 3, // Predicate_FP_ZERO
/* 18286*/     OPC_MoveParent,
/* 18287*/     OPC_MoveChild4,
/* 18288*/     OPC_Scope, 105, /*->18395*/ // 8 children in Scope
/* 18290*/      OPC_CheckCondCode, ISD::SETOEQ,
/* 18292*/      OPC_MoveParent,
/* 18293*/      OPC_CheckType, MVT::f32,
/* 18295*/      OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 18297*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18300*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18303*/      OPC_EmitInteger, MVT::i32, 1, 
/* 18306*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18309*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18312*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18315*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18318*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18321*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18324*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 18336*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18339*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18342*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18345*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 18357*/      OPC_EmitInteger, MVT::i32, 1, 
/* 18360*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 18363*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18366*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18369*/      OPC_MorphNodeTo1, TARGET_VAL(R600::SETE), 0,
                    MVT::f32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
                // Src: (selectcc:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, (fpimm:{ *:[f32] })<<P:Predicate_FP_ONE>>, (fpimm:{ *:[f32] })<<P:Predicate_FP_ZERO>>, SETOEQ:{ *:[Other] }) - Complexity = 11
                // Dst: (SETE:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1)
/* 18395*/     /*Scope*/ 105, /*->18501*/
/* 18396*/      OPC_CheckCondCode, ISD::SETEQ,
/* 18398*/      OPC_MoveParent,
/* 18399*/      OPC_CheckType, MVT::f32,
/* 18401*/      OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 18403*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18406*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18409*/      OPC_EmitInteger, MVT::i32, 1, 
/* 18412*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18415*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18418*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18421*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18424*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18427*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18430*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 18442*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18445*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18448*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18451*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 18463*/      OPC_EmitInteger, MVT::i32, 1, 
/* 18466*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 18469*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18472*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18475*/      OPC_MorphNodeTo1, TARGET_VAL(R600::SETE), 0,
                    MVT::f32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
                // Src: (selectcc:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, (fpimm:{ *:[f32] })<<P:Predicate_FP_ONE>>, (fpimm:{ *:[f32] })<<P:Predicate_FP_ZERO>>, SETEQ:{ *:[Other] }) - Complexity = 11
                // Dst: (SETE:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1)
/* 18501*/     /*Scope*/ 105, /*->18607*/
/* 18502*/      OPC_CheckCondCode, ISD::SETOGT,
/* 18504*/      OPC_MoveParent,
/* 18505*/      OPC_CheckType, MVT::f32,
/* 18507*/      OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 18509*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18512*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18515*/      OPC_EmitInteger, MVT::i32, 1, 
/* 18518*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18521*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18524*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18527*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18530*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18533*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18536*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 18548*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18551*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18554*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18557*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 18569*/      OPC_EmitInteger, MVT::i32, 1, 
/* 18572*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 18575*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18578*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18581*/      OPC_MorphNodeTo1, TARGET_VAL(R600::SGT), 0,
                    MVT::f32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
                // Src: (selectcc:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, (fpimm:{ *:[f32] })<<P:Predicate_FP_ONE>>, (fpimm:{ *:[f32] })<<P:Predicate_FP_ZERO>>, SETOGT:{ *:[Other] }) - Complexity = 11
                // Dst: (SGT:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1)
/* 18607*/     /*Scope*/ 105, /*->18713*/
/* 18608*/      OPC_CheckCondCode, ISD::SETGT,
/* 18610*/      OPC_MoveParent,
/* 18611*/      OPC_CheckType, MVT::f32,
/* 18613*/      OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 18615*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18618*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18621*/      OPC_EmitInteger, MVT::i32, 1, 
/* 18624*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18627*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18630*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18633*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18636*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18639*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18642*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 18654*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18657*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18660*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18663*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 18675*/      OPC_EmitInteger, MVT::i32, 1, 
/* 18678*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 18681*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18684*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18687*/      OPC_MorphNodeTo1, TARGET_VAL(R600::SGT), 0,
                    MVT::f32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
                // Src: (selectcc:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, (fpimm:{ *:[f32] })<<P:Predicate_FP_ONE>>, (fpimm:{ *:[f32] })<<P:Predicate_FP_ZERO>>, SETGT:{ *:[Other] }) - Complexity = 11
                // Dst: (SGT:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1)
/* 18713*/     /*Scope*/ 105, /*->18819*/
/* 18714*/      OPC_CheckCondCode, ISD::SETOGE,
/* 18716*/      OPC_MoveParent,
/* 18717*/      OPC_CheckType, MVT::f32,
/* 18719*/      OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 18721*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18724*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18727*/      OPC_EmitInteger, MVT::i32, 1, 
/* 18730*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18733*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18736*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18739*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18742*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18745*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18748*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 18760*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18763*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18766*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18769*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 18781*/      OPC_EmitInteger, MVT::i32, 1, 
/* 18784*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 18787*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18790*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18793*/      OPC_MorphNodeTo1, TARGET_VAL(R600::SGE), 0,
                    MVT::f32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
                // Src: (selectcc:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, (fpimm:{ *:[f32] })<<P:Predicate_FP_ONE>>, (fpimm:{ *:[f32] })<<P:Predicate_FP_ZERO>>, SETOGE:{ *:[Other] }) - Complexity = 11
                // Dst: (SGE:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1)
/* 18819*/     /*Scope*/ 105, /*->18925*/
/* 18820*/      OPC_CheckCondCode, ISD::SETGE,
/* 18822*/      OPC_MoveParent,
/* 18823*/      OPC_CheckType, MVT::f32,
/* 18825*/      OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 18827*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18830*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18833*/      OPC_EmitInteger, MVT::i32, 1, 
/* 18836*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18839*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18842*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18845*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18848*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18851*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18854*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 18866*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18869*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18872*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18875*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 18887*/      OPC_EmitInteger, MVT::i32, 1, 
/* 18890*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 18893*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18896*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18899*/      OPC_MorphNodeTo1, TARGET_VAL(R600::SGE), 0,
                    MVT::f32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
                // Src: (selectcc:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, (fpimm:{ *:[f32] })<<P:Predicate_FP_ONE>>, (fpimm:{ *:[f32] })<<P:Predicate_FP_ZERO>>, SETGE:{ *:[Other] }) - Complexity = 11
                // Dst: (SGE:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1)
/* 18925*/     /*Scope*/ 105, /*->19031*/
/* 18926*/      OPC_CheckCondCode, ISD::SETUNE,
/* 18928*/      OPC_MoveParent,
/* 18929*/      OPC_CheckType, MVT::f32,
/* 18931*/      OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 18933*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18936*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18939*/      OPC_EmitInteger, MVT::i32, 1, 
/* 18942*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18945*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18948*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18951*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18954*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18957*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18960*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 18972*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18975*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18978*/      OPC_EmitInteger, MVT::i32, 0, 
/* 18981*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 18993*/      OPC_EmitInteger, MVT::i32, 1, 
/* 18996*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 18999*/      OPC_EmitInteger, MVT::i32, 0, 
/* 19002*/      OPC_EmitInteger, MVT::i32, 0, 
/* 19005*/      OPC_MorphNodeTo1, TARGET_VAL(R600::SNE), 0,
                    MVT::f32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
                // Src: (selectcc:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, (fpimm:{ *:[f32] })<<P:Predicate_FP_ONE>>, (fpimm:{ *:[f32] })<<P:Predicate_FP_ZERO>>, SETUNE:{ *:[Other] }) - Complexity = 11
                // Dst: (SNE:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1)
/* 19031*/     /*Scope*/ 105, /*->19137*/
/* 19032*/      OPC_CheckCondCode, ISD::SETNE,
/* 19034*/      OPC_MoveParent,
/* 19035*/      OPC_CheckType, MVT::f32,
/* 19037*/      OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 19039*/      OPC_EmitInteger, MVT::i32, 0, 
/* 19042*/      OPC_EmitInteger, MVT::i32, 0, 
/* 19045*/      OPC_EmitInteger, MVT::i32, 1, 
/* 19048*/      OPC_EmitInteger, MVT::i32, 0, 
/* 19051*/      OPC_EmitInteger, MVT::i32, 0, 
/* 19054*/      OPC_EmitInteger, MVT::i32, 0, 
/* 19057*/      OPC_EmitInteger, MVT::i32, 0, 
/* 19060*/      OPC_EmitInteger, MVT::i32, 0, 
/* 19063*/      OPC_EmitInteger, MVT::i32, 0, 
/* 19066*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19078*/      OPC_EmitInteger, MVT::i32, 0, 
/* 19081*/      OPC_EmitInteger, MVT::i32, 0, 
/* 19084*/      OPC_EmitInteger, MVT::i32, 0, 
/* 19087*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19099*/      OPC_EmitInteger, MVT::i32, 1, 
/* 19102*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 19105*/      OPC_EmitInteger, MVT::i32, 0, 
/* 19108*/      OPC_EmitInteger, MVT::i32, 0, 
/* 19111*/      OPC_MorphNodeTo1, TARGET_VAL(R600::SNE), 0,
                    MVT::f32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
                // Src: (selectcc:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, (fpimm:{ *:[f32] })<<P:Predicate_FP_ONE>>, (fpimm:{ *:[f32] })<<P:Predicate_FP_ZERO>>, SETNE:{ *:[Other] }) - Complexity = 11
                // Dst: (SNE:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1)
/* 19137*/     0, /*End of Scope*/
/* 19138*/    0, /*End of Scope*/
/* 19139*/   /*Scope*/ 108|128,9/*1260*/, /*->20401*/
/* 19141*/    OPC_MoveChild1,
/* 19142*/    OPC_CheckOpcode, TARGET_VAL(ISD::ConstantFP),
/* 19145*/    OPC_CheckPredicate, 3, // Predicate_FP_ZERO
/* 19147*/    OPC_MoveParent,
/* 19148*/    OPC_RecordChild2, // #1 = $src1
/* 19149*/    OPC_RecordChild3, // #2 = $src2
/* 19150*/    OPC_MoveChild4,
/* 19151*/    OPC_Scope, 103, /*->19256*/ // 12 children in Scope
/* 19153*/     OPC_CheckCondCode, ISD::SETOEQ,
/* 19155*/     OPC_MoveParent,
/* 19156*/     OPC_CheckType, MVT::f32,
/* 19158*/     OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 19160*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19163*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19166*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19169*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19172*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19184*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19187*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19190*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19202*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19205*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19208*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19220*/     OPC_EmitInteger, MVT::i32, 1, 
/* 19223*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 19226*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19229*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19232*/     OPC_MorphNodeTo1, TARGET_VAL(R600::CNDE_r600), 0,
                   MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (selectcc:{ *:[f32] } f32:{ *:[f32] }:$src0, (fpimm:{ *:[f32] })<<P:Predicate_FP_ZERO>>, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2, SETOEQ:{ *:[Other] }) - Complexity = 7
               // Dst: (CNDE_r600:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2)
/* 19256*/    /*Scope*/ 103, /*->19360*/
/* 19257*/     OPC_CheckCondCode, ISD::SETEQ,
/* 19259*/     OPC_MoveParent,
/* 19260*/     OPC_CheckType, MVT::f32,
/* 19262*/     OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 19264*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19267*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19270*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19273*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19276*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19288*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19291*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19294*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19306*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19309*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19312*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19324*/     OPC_EmitInteger, MVT::i32, 1, 
/* 19327*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 19330*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19333*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19336*/     OPC_MorphNodeTo1, TARGET_VAL(R600::CNDE_r600), 0,
                   MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (selectcc:{ *:[f32] } f32:{ *:[f32] }:$src0, (fpimm:{ *:[f32] })<<P:Predicate_FP_ZERO>>, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2, SETEQ:{ *:[Other] }) - Complexity = 7
               // Dst: (CNDE_r600:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2)
/* 19360*/    /*Scope*/ 103, /*->19464*/
/* 19361*/     OPC_CheckCondCode, ISD::SETOGT,
/* 19363*/     OPC_MoveParent,
/* 19364*/     OPC_CheckType, MVT::f32,
/* 19366*/     OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 19368*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19371*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19374*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19377*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19380*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19392*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19395*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19398*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19410*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19413*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19416*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19428*/     OPC_EmitInteger, MVT::i32, 1, 
/* 19431*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 19434*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19437*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19440*/     OPC_MorphNodeTo1, TARGET_VAL(R600::CNDGT_r600), 0,
                   MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (selectcc:{ *:[f32] } f32:{ *:[f32] }:$src0, (fpimm:{ *:[f32] })<<P:Predicate_FP_ZERO>>, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2, SETOGT:{ *:[Other] }) - Complexity = 7
               // Dst: (CNDGT_r600:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2)
/* 19464*/    /*Scope*/ 103, /*->19568*/
/* 19465*/     OPC_CheckCondCode, ISD::SETGT,
/* 19467*/     OPC_MoveParent,
/* 19468*/     OPC_CheckType, MVT::f32,
/* 19470*/     OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 19472*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19475*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19478*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19481*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19484*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19496*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19499*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19502*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19514*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19517*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19520*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19532*/     OPC_EmitInteger, MVT::i32, 1, 
/* 19535*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 19538*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19541*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19544*/     OPC_MorphNodeTo1, TARGET_VAL(R600::CNDGT_r600), 0,
                   MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (selectcc:{ *:[f32] } f32:{ *:[f32] }:$src0, (fpimm:{ *:[f32] })<<P:Predicate_FP_ZERO>>, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2, SETGT:{ *:[Other] }) - Complexity = 7
               // Dst: (CNDGT_r600:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2)
/* 19568*/    /*Scope*/ 103, /*->19672*/
/* 19569*/     OPC_CheckCondCode, ISD::SETOGE,
/* 19571*/     OPC_MoveParent,
/* 19572*/     OPC_CheckType, MVT::f32,
/* 19574*/     OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 19576*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19579*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19582*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19585*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19588*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19600*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19603*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19606*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19618*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19621*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19624*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19636*/     OPC_EmitInteger, MVT::i32, 1, 
/* 19639*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 19642*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19645*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19648*/     OPC_MorphNodeTo1, TARGET_VAL(R600::CNDGE_r600), 0,
                   MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (selectcc:{ *:[f32] } f32:{ *:[f32] }:$src0, (fpimm:{ *:[f32] })<<P:Predicate_FP_ZERO>>, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2, SETOGE:{ *:[Other] }) - Complexity = 7
               // Dst: (CNDGE_r600:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2)
/* 19672*/    /*Scope*/ 103, /*->19776*/
/* 19673*/     OPC_CheckCondCode, ISD::SETGE,
/* 19675*/     OPC_MoveParent,
/* 19676*/     OPC_CheckType, MVT::f32,
/* 19678*/     OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 19680*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19683*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19686*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19689*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19692*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19704*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19707*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19710*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19722*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19725*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19728*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19740*/     OPC_EmitInteger, MVT::i32, 1, 
/* 19743*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 19746*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19749*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19752*/     OPC_MorphNodeTo1, TARGET_VAL(R600::CNDGE_r600), 0,
                   MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (selectcc:{ *:[f32] } f32:{ *:[f32] }:$src0, (fpimm:{ *:[f32] })<<P:Predicate_FP_ZERO>>, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2, SETGE:{ *:[Other] }) - Complexity = 7
               // Dst: (CNDGE_r600:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2)
/* 19776*/    /*Scope*/ 103, /*->19880*/
/* 19777*/     OPC_CheckCondCode, ISD::SETOEQ,
/* 19779*/     OPC_MoveParent,
/* 19780*/     OPC_CheckType, MVT::f32,
/* 19782*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 19784*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19787*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19790*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19793*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19796*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19808*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19811*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19814*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19826*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19829*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19832*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19844*/     OPC_EmitInteger, MVT::i32, 1, 
/* 19847*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 19850*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19853*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19856*/     OPC_MorphNodeTo1, TARGET_VAL(R600::CNDE_eg), 0,
                   MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (selectcc:{ *:[f32] } f32:{ *:[f32] }:$src0, (fpimm:{ *:[f32] })<<P:Predicate_FP_ZERO>>, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2, SETOEQ:{ *:[Other] }) - Complexity = 7
               // Dst: (CNDE_eg:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2)
/* 19880*/    /*Scope*/ 103, /*->19984*/
/* 19881*/     OPC_CheckCondCode, ISD::SETEQ,
/* 19883*/     OPC_MoveParent,
/* 19884*/     OPC_CheckType, MVT::f32,
/* 19886*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 19888*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19891*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19894*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19897*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19900*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19912*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19915*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19918*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19930*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19933*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19936*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 19948*/     OPC_EmitInteger, MVT::i32, 1, 
/* 19951*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 19954*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19957*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19960*/     OPC_MorphNodeTo1, TARGET_VAL(R600::CNDE_eg), 0,
                   MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (selectcc:{ *:[f32] } f32:{ *:[f32] }:$src0, (fpimm:{ *:[f32] })<<P:Predicate_FP_ZERO>>, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2, SETEQ:{ *:[Other] }) - Complexity = 7
               // Dst: (CNDE_eg:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2)
/* 19984*/    /*Scope*/ 103, /*->20088*/
/* 19985*/     OPC_CheckCondCode, ISD::SETOGT,
/* 19987*/     OPC_MoveParent,
/* 19988*/     OPC_CheckType, MVT::f32,
/* 19990*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 19992*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19995*/     OPC_EmitInteger, MVT::i32, 0, 
/* 19998*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20001*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20004*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20016*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20019*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20022*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20034*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20037*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20040*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20052*/     OPC_EmitInteger, MVT::i32, 1, 
/* 20055*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 20058*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20061*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20064*/     OPC_MorphNodeTo1, TARGET_VAL(R600::CNDGT_eg), 0,
                   MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (selectcc:{ *:[f32] } f32:{ *:[f32] }:$src0, (fpimm:{ *:[f32] })<<P:Predicate_FP_ZERO>>, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2, SETOGT:{ *:[Other] }) - Complexity = 7
               // Dst: (CNDGT_eg:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2)
/* 20088*/    /*Scope*/ 103, /*->20192*/
/* 20089*/     OPC_CheckCondCode, ISD::SETGT,
/* 20091*/     OPC_MoveParent,
/* 20092*/     OPC_CheckType, MVT::f32,
/* 20094*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 20096*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20099*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20102*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20105*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20108*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20120*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20123*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20126*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20138*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20141*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20144*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20156*/     OPC_EmitInteger, MVT::i32, 1, 
/* 20159*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 20162*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20165*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20168*/     OPC_MorphNodeTo1, TARGET_VAL(R600::CNDGT_eg), 0,
                   MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (selectcc:{ *:[f32] } f32:{ *:[f32] }:$src0, (fpimm:{ *:[f32] })<<P:Predicate_FP_ZERO>>, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2, SETGT:{ *:[Other] }) - Complexity = 7
               // Dst: (CNDGT_eg:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2)
/* 20192*/    /*Scope*/ 103, /*->20296*/
/* 20193*/     OPC_CheckCondCode, ISD::SETOGE,
/* 20195*/     OPC_MoveParent,
/* 20196*/     OPC_CheckType, MVT::f32,
/* 20198*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 20200*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20203*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20206*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20209*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20212*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20224*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20227*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20230*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20242*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20245*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20248*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20260*/     OPC_EmitInteger, MVT::i32, 1, 
/* 20263*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 20266*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20269*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20272*/     OPC_MorphNodeTo1, TARGET_VAL(R600::CNDGE_eg), 0,
                   MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (selectcc:{ *:[f32] } f32:{ *:[f32] }:$src0, (fpimm:{ *:[f32] })<<P:Predicate_FP_ZERO>>, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2, SETOGE:{ *:[Other] }) - Complexity = 7
               // Dst: (CNDGE_eg:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2)
/* 20296*/    /*Scope*/ 103, /*->20400*/
/* 20297*/     OPC_CheckCondCode, ISD::SETGE,
/* 20299*/     OPC_MoveParent,
/* 20300*/     OPC_CheckType, MVT::f32,
/* 20302*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 20304*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20307*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20310*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20313*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20316*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20328*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20331*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20334*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20346*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20349*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20352*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20364*/     OPC_EmitInteger, MVT::i32, 1, 
/* 20367*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 20370*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20373*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20376*/     OPC_MorphNodeTo1, TARGET_VAL(R600::CNDGE_eg), 0,
                   MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (selectcc:{ *:[f32] } f32:{ *:[f32] }:$src0, (fpimm:{ *:[f32] })<<P:Predicate_FP_ZERO>>, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2, SETGE:{ *:[Other] }) - Complexity = 7
               // Dst: (CNDGE_eg:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2)
/* 20400*/    0, /*End of Scope*/
/* 20401*/   0, /*End of Scope*/
/* 20402*/  /*Scope*/ 106|128,11/*1514*/, /*->21918*/
/* 20404*/   OPC_CheckChild0Type, MVT::i32,
/* 20406*/   OPC_Scope, 13|128,5/*653*/, /*->21062*/ // 3 children in Scope
/* 20409*/    OPC_RecordChild1, // #1 = $src1
/* 20410*/    OPC_CheckChild2Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20421*/    OPC_CheckChild3Integer, 0, 
/* 20423*/    OPC_MoveChild4,
/* 20424*/    OPC_Scope, 105, /*->20531*/ // 6 children in Scope
/* 20426*/     OPC_CheckCondCode, ISD::SETEQ,
/* 20428*/     OPC_MoveParent,
/* 20429*/     OPC_CheckType, MVT::i32,
/* 20431*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 20433*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20436*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20439*/     OPC_EmitInteger, MVT::i32, 1, 
/* 20442*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20445*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20448*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20451*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20454*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20457*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20460*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20472*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20475*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20478*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20481*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20493*/     OPC_EmitInteger, MVT::i32, 1, 
/* 20496*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 20499*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20502*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20505*/     OPC_MorphNodeTo1, TARGET_VAL(R600::SETE_INT), 0,
                   MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
               // Src: (selectcc:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, -1:{ *:[i32] }, 0:{ *:[i32] }, SETEQ:{ *:[Other] }) - Complexity = 13
               // Dst: (SETE_INT:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 20531*/    /*Scope*/ 105, /*->20637*/
/* 20532*/     OPC_CheckCondCode, ISD::SETGT,
/* 20534*/     OPC_MoveParent,
/* 20535*/     OPC_CheckType, MVT::i32,
/* 20537*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 20539*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20542*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20545*/     OPC_EmitInteger, MVT::i32, 1, 
/* 20548*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20551*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20554*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20557*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20560*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20563*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20566*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20578*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20581*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20584*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20587*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20599*/     OPC_EmitInteger, MVT::i32, 1, 
/* 20602*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 20605*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20608*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20611*/     OPC_MorphNodeTo1, TARGET_VAL(R600::SETGT_INT), 0,
                   MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
               // Src: (selectcc:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, -1:{ *:[i32] }, 0:{ *:[i32] }, SETGT:{ *:[Other] }) - Complexity = 13
               // Dst: (SETGT_INT:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 20637*/    /*Scope*/ 105, /*->20743*/
/* 20638*/     OPC_CheckCondCode, ISD::SETGE,
/* 20640*/     OPC_MoveParent,
/* 20641*/     OPC_CheckType, MVT::i32,
/* 20643*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 20645*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20648*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20651*/     OPC_EmitInteger, MVT::i32, 1, 
/* 20654*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20657*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20660*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20663*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20666*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20669*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20672*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20684*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20687*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20690*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20693*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20705*/     OPC_EmitInteger, MVT::i32, 1, 
/* 20708*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 20711*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20714*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20717*/     OPC_MorphNodeTo1, TARGET_VAL(R600::SETGE_INT), 0,
                   MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
               // Src: (selectcc:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, -1:{ *:[i32] }, 0:{ *:[i32] }, SETGE:{ *:[Other] }) - Complexity = 13
               // Dst: (SETGE_INT:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 20743*/    /*Scope*/ 105, /*->20849*/
/* 20744*/     OPC_CheckCondCode, ISD::SETNE,
/* 20746*/     OPC_MoveParent,
/* 20747*/     OPC_CheckType, MVT::i32,
/* 20749*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 20751*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20754*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20757*/     OPC_EmitInteger, MVT::i32, 1, 
/* 20760*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20763*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20766*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20769*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20772*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20775*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20778*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20790*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20793*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20796*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20799*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20811*/     OPC_EmitInteger, MVT::i32, 1, 
/* 20814*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 20817*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20820*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20823*/     OPC_MorphNodeTo1, TARGET_VAL(R600::SETNE_INT), 0,
                   MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
               // Src: (selectcc:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, -1:{ *:[i32] }, 0:{ *:[i32] }, SETNE:{ *:[Other] }) - Complexity = 13
               // Dst: (SETNE_INT:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 20849*/    /*Scope*/ 105, /*->20955*/
/* 20850*/     OPC_CheckCondCode, ISD::SETUGT,
/* 20852*/     OPC_MoveParent,
/* 20853*/     OPC_CheckType, MVT::i32,
/* 20855*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 20857*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20860*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20863*/     OPC_EmitInteger, MVT::i32, 1, 
/* 20866*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20869*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20872*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20875*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20878*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20881*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20884*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20896*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20899*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20902*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20905*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 20917*/     OPC_EmitInteger, MVT::i32, 1, 
/* 20920*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 20923*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20926*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20929*/     OPC_MorphNodeTo1, TARGET_VAL(R600::SETGT_UINT), 0,
                   MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
               // Src: (selectcc:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, -1:{ *:[i32] }, 0:{ *:[i32] }, SETUGT:{ *:[Other] }) - Complexity = 13
               // Dst: (SETGT_UINT:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 20955*/    /*Scope*/ 105, /*->21061*/
/* 20956*/     OPC_CheckCondCode, ISD::SETUGE,
/* 20958*/     OPC_MoveParent,
/* 20959*/     OPC_CheckType, MVT::i32,
/* 20961*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 20963*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20966*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20969*/     OPC_EmitInteger, MVT::i32, 1, 
/* 20972*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20975*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20978*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20981*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20984*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20987*/     OPC_EmitInteger, MVT::i32, 0, 
/* 20990*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21002*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21005*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21008*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21011*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21023*/     OPC_EmitInteger, MVT::i32, 1, 
/* 21026*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 21029*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21032*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21035*/     OPC_MorphNodeTo1, TARGET_VAL(R600::SETGE_UINT), 0,
                   MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
               // Src: (selectcc:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, -1:{ *:[i32] }, 0:{ *:[i32] }, SETUGE:{ *:[Other] }) - Complexity = 13
               // Dst: (SETGE_UINT:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 21061*/    0, /*End of Scope*/
/* 21062*/   /*Scope*/ 95|128,5/*735*/, /*->21799*/
/* 21064*/    OPC_CheckChild1Integer, 0, 
/* 21066*/    OPC_RecordChild2, // #1 = $src1
/* 21067*/    OPC_RecordChild3, // #2 = $src2
/* 21068*/    OPC_MoveChild4,
/* 21069*/    OPC_Scope, 103, /*->21174*/ // 7 children in Scope
/* 21071*/     OPC_CheckCondCode, ISD::SETEQ,
/* 21073*/     OPC_MoveParent,
/* 21074*/     OPC_CheckType, MVT::i32,
/* 21076*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 21078*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21081*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21084*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21087*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21090*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21102*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21105*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21108*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21120*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21123*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21126*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21138*/     OPC_EmitInteger, MVT::i32, 1, 
/* 21141*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 21144*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21147*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21150*/     OPC_MorphNodeTo1, TARGET_VAL(R600::CNDE_INT), 0,
                   MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (selectcc:{ *:[i32] } i32:{ *:[i32] }:$src0, 0:{ *:[i32] }, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2, SETEQ:{ *:[Other] }) - Complexity = 8
               // Dst: (CNDE_INT:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2)
/* 21174*/    /*Scope*/ 103, /*->21278*/
/* 21175*/     OPC_CheckCondCode, ISD::SETUEQ,
/* 21177*/     OPC_MoveParent,
/* 21178*/     OPC_CheckType, MVT::i32,
/* 21180*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 21182*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21185*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21188*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21191*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21194*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21206*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21209*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21212*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21224*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21227*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21230*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21242*/     OPC_EmitInteger, MVT::i32, 1, 
/* 21245*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 21248*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21251*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21254*/     OPC_MorphNodeTo1, TARGET_VAL(R600::CNDE_INT), 0,
                   MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (selectcc:{ *:[i32] } i32:{ *:[i32] }:$src0, 0:{ *:[i32] }, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2, SETUEQ:{ *:[Other] }) - Complexity = 8
               // Dst: (CNDE_INT:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2)
/* 21278*/    /*Scope*/ 103, /*->21382*/
/* 21279*/     OPC_CheckCondCode, ISD::SETGE,
/* 21281*/     OPC_MoveParent,
/* 21282*/     OPC_CheckType, MVT::i32,
/* 21284*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 21286*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21289*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21292*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21295*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21298*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21310*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21313*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21316*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21328*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21331*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21334*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21346*/     OPC_EmitInteger, MVT::i32, 1, 
/* 21349*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 21352*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21355*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21358*/     OPC_MorphNodeTo1, TARGET_VAL(R600::CNDGE_INT), 0,
                   MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (selectcc:{ *:[i32] } i32:{ *:[i32] }:$src0, 0:{ *:[i32] }, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2, SETGE:{ *:[Other] }) - Complexity = 8
               // Dst: (CNDGE_INT:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2)
/* 21382*/    /*Scope*/ 103, /*->21486*/
/* 21383*/     OPC_CheckCondCode, ISD::SETGT,
/* 21385*/     OPC_MoveParent,
/* 21386*/     OPC_CheckType, MVT::i32,
/* 21388*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 21390*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21393*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21396*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21399*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21402*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21414*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21417*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21420*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21432*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21435*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21438*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21450*/     OPC_EmitInteger, MVT::i32, 1, 
/* 21453*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 21456*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21459*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21462*/     OPC_MorphNodeTo1, TARGET_VAL(R600::CNDGT_INT), 0,
                   MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (selectcc:{ *:[i32] } i32:{ *:[i32] }:$src0, 0:{ *:[i32] }, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2, SETGT:{ *:[Other] }) - Complexity = 8
               // Dst: (CNDGT_INT:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2)
/* 21486*/    /*Scope*/ 103, /*->21590*/
/* 21487*/     OPC_CheckCondCode, ISD::SETEQ,
/* 21489*/     OPC_MoveParent,
/* 21490*/     OPC_CheckType, MVT::f32,
/* 21492*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 21494*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21497*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21500*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21503*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21506*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21518*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21521*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21524*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21536*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21539*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21542*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21554*/     OPC_EmitInteger, MVT::i32, 1, 
/* 21557*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 21560*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21563*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21566*/     OPC_MorphNodeTo1, TARGET_VAL(R600::CNDE_INT), 0,
                   MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (selectcc:{ *:[f32] } i32:{ *:[i32] }:$src0, 0:{ *:[i32] }, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2, SETEQ:{ *:[Other] }) - Complexity = 8
               // Dst: (CNDE_INT:{ *:[f32] } ?:{ *:[i32] }:$src0, ?:{ *:[f32] }:$src1, ?:{ *:[f32] }:$src2)
/* 21590*/    /*Scope*/ 103, /*->21694*/
/* 21591*/     OPC_CheckCondCode, ISD::SETGT,
/* 21593*/     OPC_MoveParent,
/* 21594*/     OPC_CheckType, MVT::f32,
/* 21596*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 21598*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21601*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21604*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21607*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21610*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21622*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21625*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21628*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21640*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21643*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21646*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21658*/     OPC_EmitInteger, MVT::i32, 1, 
/* 21661*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 21664*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21667*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21670*/     OPC_MorphNodeTo1, TARGET_VAL(R600::CNDGT_INT), 0,
                   MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (selectcc:{ *:[f32] } i32:{ *:[i32] }:$src0, 0:{ *:[i32] }, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2, SETGT:{ *:[Other] }) - Complexity = 8
               // Dst: (CNDGT_INT:{ *:[f32] } ?:{ *:[i32] }:$src0, ?:{ *:[f32] }:$src1, ?:{ *:[f32] }:$src2)
/* 21694*/    /*Scope*/ 103, /*->21798*/
/* 21695*/     OPC_CheckCondCode, ISD::SETGE,
/* 21697*/     OPC_MoveParent,
/* 21698*/     OPC_CheckType, MVT::f32,
/* 21700*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 21702*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21705*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21708*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21711*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21714*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21726*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21729*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21732*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21744*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21747*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21750*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21762*/     OPC_EmitInteger, MVT::i32, 1, 
/* 21765*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 21768*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21771*/     OPC_EmitInteger, MVT::i32, 0, 
/* 21774*/     OPC_MorphNodeTo1, TARGET_VAL(R600::CNDGE_INT), 0,
                   MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (selectcc:{ *:[f32] } i32:{ *:[i32] }:$src0, 0:{ *:[i32] }, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2, SETGE:{ *:[Other] }) - Complexity = 8
               // Dst: (CNDGE_INT:{ *:[f32] } ?:{ *:[i32] }:$src0, ?:{ *:[f32] }:$src1, ?:{ *:[f32] }:$src2)
/* 21798*/    0, /*End of Scope*/
/* 21799*/   /*Scope*/ 117, /*->21917*/
/* 21800*/    OPC_CheckChild1Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21811*/    OPC_RecordChild2, // #1 = $src1
/* 21812*/    OPC_RecordChild3, // #2 = $src2
/* 21813*/    OPC_MoveChild4,
/* 21814*/    OPC_CheckCondCode, ISD::SETGT,
/* 21816*/    OPC_MoveParent,
/* 21817*/    OPC_CheckType, MVT::i32,
/* 21819*/    OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 21821*/    OPC_EmitInteger, MVT::i32, 0, 
/* 21824*/    OPC_EmitInteger, MVT::i32, 0, 
/* 21827*/    OPC_EmitInteger, MVT::i32, 0, 
/* 21830*/    OPC_EmitInteger, MVT::i32, 0, 
/* 21833*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21845*/    OPC_EmitInteger, MVT::i32, 0, 
/* 21848*/    OPC_EmitInteger, MVT::i32, 0, 
/* 21851*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21863*/    OPC_EmitInteger, MVT::i32, 0, 
/* 21866*/    OPC_EmitInteger, MVT::i32, 0, 
/* 21869*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 21881*/    OPC_EmitInteger, MVT::i32, 1, 
/* 21884*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 21887*/    OPC_EmitInteger, MVT::i32, 0, 
/* 21890*/    OPC_EmitInteger, MVT::i32, 0, 
/* 21893*/    OPC_MorphNodeTo1, TARGET_VAL(R600::CNDGE_INT), 0,
                  MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
              // Src: (selectcc:{ *:[i32] } i32:{ *:[i32] }:$src0, -1:{ *:[i32] }, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2, SETGT:{ *:[Other] }) - Complexity = 8
              // Dst: (CNDGE_INT:{ *:[i32] } ?:{ *:[i32] }:$src0, ?:{ *:[i32] }:$src1, ?:{ *:[i32] }:$src2)
/* 21917*/   0, /*End of Scope*/
/* 21918*/  0, /*End of Scope*/
/* 21919*/ /*SwitchOpcode*/ 96|128,7/*992*/, TARGET_VAL(ISD::LOAD),// ->22915
/* 21923*/  OPC_RecordMemRef,
/* 21924*/  OPC_RecordNode, // #0 = 'ld' chained node
/* 21925*/  OPC_RecordChild1, // #1 = $addr
/* 21926*/  OPC_CheckPredicate, 4, // Predicate_unindexedload
/* 21928*/  OPC_Scope, 22, /*->21952*/ // 4 children in Scope
/* 21930*/   OPC_CheckPredicate, 5, // Predicate_load_private
/* 21932*/   OPC_CheckType, MVT::i32,
/* 21934*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 21936*/   OPC_CheckComplexPat, /*CP*/0, /*#*/1, // SelectADDRIndirect:$addr #2 #3
/* 21939*/   OPC_EmitMergeInputChains1_0,
/* 21940*/   OPC_EmitInteger, MVT::i32, 0, 
/* 21943*/   OPC_MorphNodeTo1, TARGET_VAL(R600::R600_RegisterLoad), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 3/*#Ops*/, 2, 3, 4, 
             // Src: (ld:{ *:[i32] } ADDRIndirect:{ *:[iPTR] }:$addr)<<P:Predicate_unindexedload>><<P:Predicate_load_private>> - Complexity = 13
             // Dst: (R600_RegisterLoad:{ *:[i32] } FRAMEri:{ *:[iPTR] }:$addr, 0:{ *:[i32] })
/* 21952*/  /*Scope*/ 66|128,3/*450*/, /*->22404*/
/* 21954*/   OPC_CheckChild1Type, MVT::i32,
/* 21956*/   OPC_CheckType, MVT::i32,
/* 21958*/   OPC_Scope, 50, /*->22010*/ // 12 children in Scope
/* 21960*/    OPC_CheckPredicate, 6, // Predicate_az_extload
/* 21962*/    OPC_Scope, 22, /*->21986*/ // 2 children in Scope
/* 21964*/     OPC_CheckPredicate, 7, // Predicate_az_extloadi8
/* 21966*/     OPC_CheckPredicate, 8, // Predicate_vtx_id3_az_extloadi8
/* 21968*/     OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 21970*/     OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 21973*/     OPC_EmitMergeInputChains1_0,
/* 21974*/     OPC_EmitInteger, MVT::i8, 3, 
/* 21977*/     OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_8_eg), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 3/*#Ops*/, 2, 3, 4, 
               // Src: (ld:{ *:[i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_az_extload>><<P:Predicate_az_extloadi8>><<P:Predicate_vtx_id3_az_extloadi8>> - Complexity = 13
               // Dst: (VTX_READ_8_eg:{ *:[i32] } MEMxi:{ *:[i32] }:$src_gpr, 3:{ *:[i8] })
/* 21986*/    /*Scope*/ 22, /*->22009*/
/* 21987*/     OPC_CheckPredicate, 9, // Predicate_az_extloadi16
/* 21989*/     OPC_CheckPredicate, 8, // Predicate_vtx_id3_az_extloadi16
/* 21991*/     OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 21993*/     OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 21996*/     OPC_EmitMergeInputChains1_0,
/* 21997*/     OPC_EmitInteger, MVT::i8, 3, 
/* 22000*/     OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_16_eg), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 3/*#Ops*/, 2, 3, 4, 
               // Src: (ld:{ *:[i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_az_extload>><<P:Predicate_az_extloadi16>><<P:Predicate_vtx_id3_az_extloadi16>> - Complexity = 13
               // Dst: (VTX_READ_16_eg:{ *:[i32] } MEMxi:{ *:[i32] }:$src_gpr, 3:{ *:[i8] })
/* 22009*/    0, /*End of Scope*/
/* 22010*/   /*Scope*/ 22, /*->22033*/
/* 22011*/    OPC_CheckPredicate, 10, // Predicate_load
/* 22013*/    OPC_CheckPredicate, 8, // Predicate_vtx_id3_load
/* 22015*/    OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 22017*/    OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22020*/    OPC_EmitMergeInputChains1_0,
/* 22021*/    OPC_EmitInteger, MVT::i8, 3, 
/* 22024*/    OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_32_eg), 0|OPFL_Chain|OPFL_MemRefs,
                  MVT::i32, 3/*#Ops*/, 2, 3, 4, 
              // Src: (ld:{ *:[i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_load>><<P:Predicate_vtx_id3_load>> - Complexity = 13
              // Dst: (VTX_READ_32_eg:{ *:[i32] } MEMxi:{ *:[i32] }:$src_gpr, 3:{ *:[i8] })
/* 22033*/   /*Scope*/ 50, /*->22084*/
/* 22034*/    OPC_CheckPredicate, 6, // Predicate_az_extload
/* 22036*/    OPC_Scope, 22, /*->22060*/ // 2 children in Scope
/* 22038*/     OPC_CheckPredicate, 7, // Predicate_az_extloadi8
/* 22040*/     OPC_CheckPredicate, 11, // Predicate_vtx_id2_az_extloadi8
/* 22042*/     OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 22044*/     OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22047*/     OPC_EmitMergeInputChains1_0,
/* 22048*/     OPC_EmitInteger, MVT::i8, 2, 
/* 22051*/     OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_8_eg), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 3/*#Ops*/, 2, 3, 4, 
               // Src: (ld:{ *:[i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_az_extload>><<P:Predicate_az_extloadi8>><<P:Predicate_vtx_id2_az_extloadi8>> - Complexity = 13
               // Dst: (VTX_READ_8_eg:{ *:[i32] } MEMxi:{ *:[i32] }:$src_gpr, 2:{ *:[i8] })
/* 22060*/    /*Scope*/ 22, /*->22083*/
/* 22061*/     OPC_CheckPredicate, 9, // Predicate_az_extloadi16
/* 22063*/     OPC_CheckPredicate, 11, // Predicate_vtx_id2_az_extloadi16
/* 22065*/     OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 22067*/     OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22070*/     OPC_EmitMergeInputChains1_0,
/* 22071*/     OPC_EmitInteger, MVT::i8, 2, 
/* 22074*/     OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_16_eg), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 3/*#Ops*/, 2, 3, 4, 
               // Src: (ld:{ *:[i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_az_extload>><<P:Predicate_az_extloadi16>><<P:Predicate_vtx_id2_az_extloadi16>> - Complexity = 13
               // Dst: (VTX_READ_16_eg:{ *:[i32] } MEMxi:{ *:[i32] }:$src_gpr, 2:{ *:[i8] })
/* 22083*/    0, /*End of Scope*/
/* 22084*/   /*Scope*/ 22, /*->22107*/
/* 22085*/    OPC_CheckPredicate, 10, // Predicate_load
/* 22087*/    OPC_CheckPredicate, 11, // Predicate_vtx_id2_load
/* 22089*/    OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 22091*/    OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22094*/    OPC_EmitMergeInputChains1_0,
/* 22095*/    OPC_EmitInteger, MVT::i8, 2, 
/* 22098*/    OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_32_eg), 0|OPFL_Chain|OPFL_MemRefs,
                  MVT::i32, 3/*#Ops*/, 2, 3, 4, 
              // Src: (ld:{ *:[i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_load>><<P:Predicate_vtx_id2_load>> - Complexity = 13
              // Dst: (VTX_READ_32_eg:{ *:[i32] } MEMxi:{ *:[i32] }:$src_gpr, 2:{ *:[i8] })
/* 22107*/   /*Scope*/ 50, /*->22158*/
/* 22108*/    OPC_CheckPredicate, 6, // Predicate_az_extload
/* 22110*/    OPC_Scope, 22, /*->22134*/ // 2 children in Scope
/* 22112*/     OPC_CheckPredicate, 7, // Predicate_az_extloadi8
/* 22114*/     OPC_CheckPredicate, 12, // Predicate_vtx_id1_az_extloadi8
/* 22116*/     OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 22118*/     OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22121*/     OPC_EmitMergeInputChains1_0,
/* 22122*/     OPC_EmitInteger, MVT::i8, 1, 
/* 22125*/     OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_8_eg), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 3/*#Ops*/, 2, 3, 4, 
               // Src: (ld:{ *:[i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_az_extload>><<P:Predicate_az_extloadi8>><<P:Predicate_vtx_id1_az_extloadi8>> - Complexity = 13
               // Dst: (VTX_READ_8_eg:{ *:[i32] } MEMxi:{ *:[i32] }:$src_gpr, 1:{ *:[i8] })
/* 22134*/    /*Scope*/ 22, /*->22157*/
/* 22135*/     OPC_CheckPredicate, 9, // Predicate_az_extloadi16
/* 22137*/     OPC_CheckPredicate, 12, // Predicate_vtx_id1_az_extloadi16
/* 22139*/     OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 22141*/     OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22144*/     OPC_EmitMergeInputChains1_0,
/* 22145*/     OPC_EmitInteger, MVT::i8, 1, 
/* 22148*/     OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_16_eg), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 3/*#Ops*/, 2, 3, 4, 
               // Src: (ld:{ *:[i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_az_extload>><<P:Predicate_az_extloadi16>><<P:Predicate_vtx_id1_az_extloadi16>> - Complexity = 13
               // Dst: (VTX_READ_16_eg:{ *:[i32] } MEMxi:{ *:[i32] }:$src_gpr, 1:{ *:[i8] })
/* 22157*/    0, /*End of Scope*/
/* 22158*/   /*Scope*/ 22, /*->22181*/
/* 22159*/    OPC_CheckPredicate, 10, // Predicate_load
/* 22161*/    OPC_CheckPredicate, 12, // Predicate_vtx_id1_load
/* 22163*/    OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 22165*/    OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22168*/    OPC_EmitMergeInputChains1_0,
/* 22169*/    OPC_EmitInteger, MVT::i8, 1, 
/* 22172*/    OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_32_eg), 0|OPFL_Chain|OPFL_MemRefs,
                  MVT::i32, 3/*#Ops*/, 2, 3, 4, 
              // Src: (ld:{ *:[i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_load>><<P:Predicate_vtx_id1_load>> - Complexity = 13
              // Dst: (VTX_READ_32_eg:{ *:[i32] } MEMxi:{ *:[i32] }:$src_gpr, 1:{ *:[i8] })
/* 22181*/   /*Scope*/ 50, /*->22232*/
/* 22182*/    OPC_CheckPredicate, 6, // Predicate_az_extload
/* 22184*/    OPC_Scope, 22, /*->22208*/ // 2 children in Scope
/* 22186*/     OPC_CheckPredicate, 7, // Predicate_az_extloadi8
/* 22188*/     OPC_CheckPredicate, 8, // Predicate_vtx_id3_az_extloadi8
/* 22190*/     OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 22192*/     OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22195*/     OPC_EmitMergeInputChains1_0,
/* 22196*/     OPC_EmitInteger, MVT::i8, 3, 
/* 22199*/     OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_8_cm), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 3/*#Ops*/, 2, 3, 4, 
               // Src: (ld:{ *:[i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_az_extload>><<P:Predicate_az_extloadi8>><<P:Predicate_vtx_id3_az_extloadi8>> - Complexity = 13
               // Dst: (VTX_READ_8_cm:{ *:[i32] } MEMxi:{ *:[i32] }:$src_gpr, 3:{ *:[i8] })
/* 22208*/    /*Scope*/ 22, /*->22231*/
/* 22209*/     OPC_CheckPredicate, 9, // Predicate_az_extloadi16
/* 22211*/     OPC_CheckPredicate, 8, // Predicate_vtx_id3_az_extloadi16
/* 22213*/     OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 22215*/     OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22218*/     OPC_EmitMergeInputChains1_0,
/* 22219*/     OPC_EmitInteger, MVT::i8, 3, 
/* 22222*/     OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_16_cm), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 3/*#Ops*/, 2, 3, 4, 
               // Src: (ld:{ *:[i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_az_extload>><<P:Predicate_az_extloadi16>><<P:Predicate_vtx_id3_az_extloadi16>> - Complexity = 13
               // Dst: (VTX_READ_16_cm:{ *:[i32] } MEMxi:{ *:[i32] }:$src_gpr, 3:{ *:[i8] })
/* 22231*/    0, /*End of Scope*/
/* 22232*/   /*Scope*/ 22, /*->22255*/
/* 22233*/    OPC_CheckPredicate, 10, // Predicate_load
/* 22235*/    OPC_CheckPredicate, 8, // Predicate_vtx_id3_load
/* 22237*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 22239*/    OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22242*/    OPC_EmitMergeInputChains1_0,
/* 22243*/    OPC_EmitInteger, MVT::i8, 3, 
/* 22246*/    OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_32_cm), 0|OPFL_Chain|OPFL_MemRefs,
                  MVT::i32, 3/*#Ops*/, 2, 3, 4, 
              // Src: (ld:{ *:[i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_load>><<P:Predicate_vtx_id3_load>> - Complexity = 13
              // Dst: (VTX_READ_32_cm:{ *:[i32] } MEMxi:{ *:[i32] }:$src_gpr, 3:{ *:[i8] })
/* 22255*/   /*Scope*/ 50, /*->22306*/
/* 22256*/    OPC_CheckPredicate, 6, // Predicate_az_extload
/* 22258*/    OPC_Scope, 22, /*->22282*/ // 2 children in Scope
/* 22260*/     OPC_CheckPredicate, 7, // Predicate_az_extloadi8
/* 22262*/     OPC_CheckPredicate, 11, // Predicate_vtx_id2_az_extloadi8
/* 22264*/     OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 22266*/     OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22269*/     OPC_EmitMergeInputChains1_0,
/* 22270*/     OPC_EmitInteger, MVT::i8, 2, 
/* 22273*/     OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_8_cm), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 3/*#Ops*/, 2, 3, 4, 
               // Src: (ld:{ *:[i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_az_extload>><<P:Predicate_az_extloadi8>><<P:Predicate_vtx_id2_az_extloadi8>> - Complexity = 13
               // Dst: (VTX_READ_8_cm:{ *:[i32] } MEMxi:{ *:[i32] }:$src_gpr, 2:{ *:[i8] })
/* 22282*/    /*Scope*/ 22, /*->22305*/
/* 22283*/     OPC_CheckPredicate, 9, // Predicate_az_extloadi16
/* 22285*/     OPC_CheckPredicate, 11, // Predicate_vtx_id2_az_extloadi16
/* 22287*/     OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 22289*/     OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22292*/     OPC_EmitMergeInputChains1_0,
/* 22293*/     OPC_EmitInteger, MVT::i8, 2, 
/* 22296*/     OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_16_cm), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 3/*#Ops*/, 2, 3, 4, 
               // Src: (ld:{ *:[i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_az_extload>><<P:Predicate_az_extloadi16>><<P:Predicate_vtx_id2_az_extloadi16>> - Complexity = 13
               // Dst: (VTX_READ_16_cm:{ *:[i32] } MEMxi:{ *:[i32] }:$src_gpr, 2:{ *:[i8] })
/* 22305*/    0, /*End of Scope*/
/* 22306*/   /*Scope*/ 22, /*->22329*/
/* 22307*/    OPC_CheckPredicate, 10, // Predicate_load
/* 22309*/    OPC_CheckPredicate, 11, // Predicate_vtx_id2_load
/* 22311*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 22313*/    OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22316*/    OPC_EmitMergeInputChains1_0,
/* 22317*/    OPC_EmitInteger, MVT::i8, 2, 
/* 22320*/    OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_32_cm), 0|OPFL_Chain|OPFL_MemRefs,
                  MVT::i32, 3/*#Ops*/, 2, 3, 4, 
              // Src: (ld:{ *:[i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_load>><<P:Predicate_vtx_id2_load>> - Complexity = 13
              // Dst: (VTX_READ_32_cm:{ *:[i32] } MEMxi:{ *:[i32] }:$src_gpr, 2:{ *:[i8] })
/* 22329*/   /*Scope*/ 50, /*->22380*/
/* 22330*/    OPC_CheckPredicate, 6, // Predicate_az_extload
/* 22332*/    OPC_Scope, 22, /*->22356*/ // 2 children in Scope
/* 22334*/     OPC_CheckPredicate, 7, // Predicate_az_extloadi8
/* 22336*/     OPC_CheckPredicate, 12, // Predicate_vtx_id1_az_extloadi8
/* 22338*/     OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 22340*/     OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22343*/     OPC_EmitMergeInputChains1_0,
/* 22344*/     OPC_EmitInteger, MVT::i8, 1, 
/* 22347*/     OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_8_cm), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 3/*#Ops*/, 2, 3, 4, 
               // Src: (ld:{ *:[i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_az_extload>><<P:Predicate_az_extloadi8>><<P:Predicate_vtx_id1_az_extloadi8>> - Complexity = 13
               // Dst: (VTX_READ_8_cm:{ *:[i32] } MEMxi:{ *:[i32] }:$src_gpr, 1:{ *:[i8] })
/* 22356*/    /*Scope*/ 22, /*->22379*/
/* 22357*/     OPC_CheckPredicate, 9, // Predicate_az_extloadi16
/* 22359*/     OPC_CheckPredicate, 12, // Predicate_vtx_id1_az_extloadi16
/* 22361*/     OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 22363*/     OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22366*/     OPC_EmitMergeInputChains1_0,
/* 22367*/     OPC_EmitInteger, MVT::i8, 1, 
/* 22370*/     OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_16_cm), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 3/*#Ops*/, 2, 3, 4, 
               // Src: (ld:{ *:[i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_az_extload>><<P:Predicate_az_extloadi16>><<P:Predicate_vtx_id1_az_extloadi16>> - Complexity = 13
               // Dst: (VTX_READ_16_cm:{ *:[i32] } MEMxi:{ *:[i32] }:$src_gpr, 1:{ *:[i8] })
/* 22379*/    0, /*End of Scope*/
/* 22380*/   /*Scope*/ 22, /*->22403*/
/* 22381*/    OPC_CheckPredicate, 10, // Predicate_load
/* 22383*/    OPC_CheckPredicate, 12, // Predicate_vtx_id1_load
/* 22385*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 22387*/    OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22390*/    OPC_EmitMergeInputChains1_0,
/* 22391*/    OPC_EmitInteger, MVT::i8, 1, 
/* 22394*/    OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_32_cm), 0|OPFL_Chain|OPFL_MemRefs,
                  MVT::i32, 3/*#Ops*/, 2, 3, 4, 
              // Src: (ld:{ *:[i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_load>><<P:Predicate_vtx_id1_load>> - Complexity = 13
              // Dst: (VTX_READ_32_cm:{ *:[i32] } MEMxi:{ *:[i32] }:$src_gpr, 1:{ *:[i8] })
/* 22403*/   0, /*End of Scope*/
/* 22404*/  /*Scope*/ 43, /*->22448*/
/* 22405*/   OPC_CheckPredicate, 13, // Predicate_load_local
/* 22407*/   OPC_CheckType, MVT::i32,
/* 22409*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 22411*/   OPC_EmitMergeInputChains1_0,
/* 22412*/   OPC_EmitInteger, MVT::i32, 0, 
/* 22415*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 22427*/   OPC_EmitInteger, MVT::i32, 1, 
/* 22430*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 22433*/   OPC_EmitInteger, MVT::i32, 0, 
/* 22436*/   OPC_MorphNodeTo1, TARGET_VAL(R600::LDS_READ_RET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 6/*#Ops*/, 1, 2, 3, 4, 5, 6, 
             // Src: (ld:{ *:[i32] } R600_Reg32:{ *:[iPTR] }:$src0)<<P:Predicate_unindexedload>><<P:Predicate_load_local>> - Complexity = 4
             // Dst: (LDS_READ_RET:{ *:[i32] } R600_Reg32:{ *:[iPTR] }:$src0)
/* 22448*/  /*Scope*/ 80|128,3/*464*/, /*->22914*/
/* 22450*/   OPC_CheckChild1Type, MVT::i32,
/* 22452*/   OPC_Scope, 45, /*->22499*/ // 5 children in Scope
/* 22454*/    OPC_CheckPredicate, 14, // Predicate_sextload
/* 22456*/    OPC_CheckPredicate, 15, // Predicate_sextloadi8_local
/* 22458*/    OPC_CheckType, MVT::i32,
/* 22460*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 22462*/    OPC_EmitMergeInputChains1_0,
/* 22463*/    OPC_EmitInteger, MVT::i32, 0, 
/* 22466*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 22478*/    OPC_EmitInteger, MVT::i32, 1, 
/* 22481*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 22484*/    OPC_EmitInteger, MVT::i32, 0, 
/* 22487*/    OPC_MorphNodeTo1, TARGET_VAL(R600::LDS_BYTE_READ_RET), 0|OPFL_Chain|OPFL_MemRefs,
                  MVT::i32, 6/*#Ops*/, 1, 2, 3, 4, 5, 6, 
              // Src: (ld:{ *:[i32] } i32:{ *:[i32] }:$src0)<<P:Predicate_unindexedload>><<P:Predicate_sextload>><<P:Predicate_sextloadi8_local>> - Complexity = 4
              // Dst: (LDS_BYTE_READ_RET:{ *:[i32] } i32:{ *:[i32] }:$src0)
/* 22499*/   /*Scope*/ 45, /*->22545*/
/* 22500*/    OPC_CheckPredicate, 6, // Predicate_az_extload
/* 22502*/    OPC_CheckPredicate, 7, // Predicate_az_extloadi8
/* 22504*/    OPC_CheckType, MVT::i32,
/* 22506*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 22508*/    OPC_EmitMergeInputChains1_0,
/* 22509*/    OPC_EmitInteger, MVT::i32, 0, 
/* 22512*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 22524*/    OPC_EmitInteger, MVT::i32, 1, 
/* 22527*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 22530*/    OPC_EmitInteger, MVT::i32, 0, 
/* 22533*/    OPC_MorphNodeTo1, TARGET_VAL(R600::LDS_UBYTE_READ_RET), 0|OPFL_Chain|OPFL_MemRefs,
                  MVT::i32, 6/*#Ops*/, 1, 2, 3, 4, 5, 6, 
              // Src: (ld:{ *:[i32] } i32:{ *:[i32] }:$src0)<<P:Predicate_unindexedload>><<P:Predicate_az_extload>><<P:Predicate_az_extloadi8>> - Complexity = 4
              // Dst: (LDS_UBYTE_READ_RET:{ *:[i32] } i32:{ *:[i32] }:$src0)
/* 22545*/   /*Scope*/ 45, /*->22591*/
/* 22546*/    OPC_CheckPredicate, 14, // Predicate_sextload
/* 22548*/    OPC_CheckPredicate, 16, // Predicate_sextloadi16_local
/* 22550*/    OPC_CheckType, MVT::i32,
/* 22552*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 22554*/    OPC_EmitMergeInputChains1_0,
/* 22555*/    OPC_EmitInteger, MVT::i32, 0, 
/* 22558*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 22570*/    OPC_EmitInteger, MVT::i32, 1, 
/* 22573*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 22576*/    OPC_EmitInteger, MVT::i32, 0, 
/* 22579*/    OPC_MorphNodeTo1, TARGET_VAL(R600::LDS_SHORT_READ_RET), 0|OPFL_Chain|OPFL_MemRefs,
                  MVT::i32, 6/*#Ops*/, 1, 2, 3, 4, 5, 6, 
              // Src: (ld:{ *:[i32] } i32:{ *:[i32] }:$src0)<<P:Predicate_unindexedload>><<P:Predicate_sextload>><<P:Predicate_sextloadi16_local>> - Complexity = 4
              // Dst: (LDS_SHORT_READ_RET:{ *:[i32] } i32:{ *:[i32] }:$src0)
/* 22591*/   /*Scope*/ 45, /*->22637*/
/* 22592*/    OPC_CheckPredicate, 6, // Predicate_az_extload
/* 22594*/    OPC_CheckPredicate, 9, // Predicate_az_extloadi16
/* 22596*/    OPC_CheckType, MVT::i32,
/* 22598*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 22600*/    OPC_EmitMergeInputChains1_0,
/* 22601*/    OPC_EmitInteger, MVT::i32, 0, 
/* 22604*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 22616*/    OPC_EmitInteger, MVT::i32, 1, 
/* 22619*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 22622*/    OPC_EmitInteger, MVT::i32, 0, 
/* 22625*/    OPC_MorphNodeTo1, TARGET_VAL(R600::LDS_USHORT_READ_RET), 0|OPFL_Chain|OPFL_MemRefs,
                  MVT::i32, 6/*#Ops*/, 1, 2, 3, 4, 5, 6, 
              // Src: (ld:{ *:[i32] } i32:{ *:[i32] }:$src0)<<P:Predicate_unindexedload>><<P:Predicate_az_extload>><<P:Predicate_az_extloadi16>> - Complexity = 4
              // Dst: (LDS_USHORT_READ_RET:{ *:[i32] } i32:{ *:[i32] }:$src0)
/* 22637*/   /*Scope*/ 18|128,2/*274*/, /*->22913*/
/* 22639*/    OPC_CheckPredicate, 10, // Predicate_load
/* 22641*/    OPC_Scope, 44, /*->22687*/ // 6 children in Scope
/* 22643*/     OPC_CheckPredicate, 8, // Predicate_vtx_id3_load
/* 22645*/     OPC_SwitchType /*2 cases */, 18, MVT::v2i32,// ->22666
/* 22648*/      OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 22650*/      OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22653*/      OPC_EmitMergeInputChains1_0,
/* 22654*/      OPC_EmitInteger, MVT::i8, 3, 
/* 22657*/      OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_64_eg), 0|OPFL_Chain|OPFL_MemRefs,
                    MVT::v2i32, 3/*#Ops*/, 2, 3, 4, 
                // Src: (ld:{ *:[v2i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_load>><<P:Predicate_vtx_id3_load>> - Complexity = 13
                // Dst: (VTX_READ_64_eg:{ *:[v2i32] } MEMxi:{ *:[i32] }:$src_gpr, 3:{ *:[i8] })
/* 22666*/     /*SwitchType*/ 18, MVT::v4i32,// ->22686
/* 22668*/      OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 22670*/      OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22673*/      OPC_EmitMergeInputChains1_0,
/* 22674*/      OPC_EmitInteger, MVT::i8, 3, 
/* 22677*/      OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_128_eg), 0|OPFL_Chain|OPFL_MemRefs,
                    MVT::v4i32, 3/*#Ops*/, 2, 3, 4, 
                // Src: (ld:{ *:[v4i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_load>><<P:Predicate_vtx_id3_load>> - Complexity = 13
                // Dst: (VTX_READ_128_eg:{ *:[v4i32] } MEMxi:{ *:[i32] }:$src_gpr, 3:{ *:[i8] })
/* 22686*/     0, // EndSwitchType
/* 22687*/    /*Scope*/ 44, /*->22732*/
/* 22688*/     OPC_CheckPredicate, 11, // Predicate_vtx_id2_load
/* 22690*/     OPC_SwitchType /*2 cases */, 18, MVT::v2i32,// ->22711
/* 22693*/      OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 22695*/      OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22698*/      OPC_EmitMergeInputChains1_0,
/* 22699*/      OPC_EmitInteger, MVT::i8, 2, 
/* 22702*/      OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_64_eg), 0|OPFL_Chain|OPFL_MemRefs,
                    MVT::v2i32, 3/*#Ops*/, 2, 3, 4, 
                // Src: (ld:{ *:[v2i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_load>><<P:Predicate_vtx_id2_load>> - Complexity = 13
                // Dst: (VTX_READ_64_eg:{ *:[v2i32] } MEMxi:{ *:[i32] }:$src_gpr, 2:{ *:[i8] })
/* 22711*/     /*SwitchType*/ 18, MVT::v4i32,// ->22731
/* 22713*/      OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 22715*/      OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22718*/      OPC_EmitMergeInputChains1_0,
/* 22719*/      OPC_EmitInteger, MVT::i8, 2, 
/* 22722*/      OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_128_eg), 0|OPFL_Chain|OPFL_MemRefs,
                    MVT::v4i32, 3/*#Ops*/, 2, 3, 4, 
                // Src: (ld:{ *:[v4i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_load>><<P:Predicate_vtx_id2_load>> - Complexity = 13
                // Dst: (VTX_READ_128_eg:{ *:[v4i32] } MEMxi:{ *:[i32] }:$src_gpr, 2:{ *:[i8] })
/* 22731*/     0, // EndSwitchType
/* 22732*/    /*Scope*/ 44, /*->22777*/
/* 22733*/     OPC_CheckPredicate, 12, // Predicate_vtx_id1_load
/* 22735*/     OPC_SwitchType /*2 cases */, 18, MVT::v2i32,// ->22756
/* 22738*/      OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 22740*/      OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22743*/      OPC_EmitMergeInputChains1_0,
/* 22744*/      OPC_EmitInteger, MVT::i8, 1, 
/* 22747*/      OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_64_eg), 0|OPFL_Chain|OPFL_MemRefs,
                    MVT::v2i32, 3/*#Ops*/, 2, 3, 4, 
                // Src: (ld:{ *:[v2i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_load>><<P:Predicate_vtx_id1_load>> - Complexity = 13
                // Dst: (VTX_READ_64_eg:{ *:[v2i32] } MEMxi:{ *:[i32] }:$src_gpr, 1:{ *:[i8] })
/* 22756*/     /*SwitchType*/ 18, MVT::v4i32,// ->22776
/* 22758*/      OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 22760*/      OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22763*/      OPC_EmitMergeInputChains1_0,
/* 22764*/      OPC_EmitInteger, MVT::i8, 1, 
/* 22767*/      OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_128_eg), 0|OPFL_Chain|OPFL_MemRefs,
                    MVT::v4i32, 3/*#Ops*/, 2, 3, 4, 
                // Src: (ld:{ *:[v4i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_load>><<P:Predicate_vtx_id1_load>> - Complexity = 13
                // Dst: (VTX_READ_128_eg:{ *:[v4i32] } MEMxi:{ *:[i32] }:$src_gpr, 1:{ *:[i8] })
/* 22776*/     0, // EndSwitchType
/* 22777*/    /*Scope*/ 44, /*->22822*/
/* 22778*/     OPC_CheckPredicate, 8, // Predicate_vtx_id3_load
/* 22780*/     OPC_SwitchType /*2 cases */, 18, MVT::v2i32,// ->22801
/* 22783*/      OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 22785*/      OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22788*/      OPC_EmitMergeInputChains1_0,
/* 22789*/      OPC_EmitInteger, MVT::i8, 3, 
/* 22792*/      OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_64_cm), 0|OPFL_Chain|OPFL_MemRefs,
                    MVT::v2i32, 3/*#Ops*/, 2, 3, 4, 
                // Src: (ld:{ *:[v2i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_load>><<P:Predicate_vtx_id3_load>> - Complexity = 13
                // Dst: (VTX_READ_64_cm:{ *:[v2i32] } MEMxi:{ *:[i32] }:$src_gpr, 3:{ *:[i8] })
/* 22801*/     /*SwitchType*/ 18, MVT::v4i32,// ->22821
/* 22803*/      OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 22805*/      OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22808*/      OPC_EmitMergeInputChains1_0,
/* 22809*/      OPC_EmitInteger, MVT::i8, 3, 
/* 22812*/      OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_128_cm), 0|OPFL_Chain|OPFL_MemRefs,
                    MVT::v4i32, 3/*#Ops*/, 2, 3, 4, 
                // Src: (ld:{ *:[v4i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_load>><<P:Predicate_vtx_id3_load>> - Complexity = 13
                // Dst: (VTX_READ_128_cm:{ *:[v4i32] } MEMxi:{ *:[i32] }:$src_gpr, 3:{ *:[i8] })
/* 22821*/     0, // EndSwitchType
/* 22822*/    /*Scope*/ 44, /*->22867*/
/* 22823*/     OPC_CheckPredicate, 11, // Predicate_vtx_id2_load
/* 22825*/     OPC_SwitchType /*2 cases */, 18, MVT::v2i32,// ->22846
/* 22828*/      OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 22830*/      OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22833*/      OPC_EmitMergeInputChains1_0,
/* 22834*/      OPC_EmitInteger, MVT::i8, 2, 
/* 22837*/      OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_64_cm), 0|OPFL_Chain|OPFL_MemRefs,
                    MVT::v2i32, 3/*#Ops*/, 2, 3, 4, 
                // Src: (ld:{ *:[v2i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_load>><<P:Predicate_vtx_id2_load>> - Complexity = 13
                // Dst: (VTX_READ_64_cm:{ *:[v2i32] } MEMxi:{ *:[i32] }:$src_gpr, 2:{ *:[i8] })
/* 22846*/     /*SwitchType*/ 18, MVT::v4i32,// ->22866
/* 22848*/      OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 22850*/      OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22853*/      OPC_EmitMergeInputChains1_0,
/* 22854*/      OPC_EmitInteger, MVT::i8, 2, 
/* 22857*/      OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_128_cm), 0|OPFL_Chain|OPFL_MemRefs,
                    MVT::v4i32, 3/*#Ops*/, 2, 3, 4, 
                // Src: (ld:{ *:[v4i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_load>><<P:Predicate_vtx_id2_load>> - Complexity = 13
                // Dst: (VTX_READ_128_cm:{ *:[v4i32] } MEMxi:{ *:[i32] }:$src_gpr, 2:{ *:[i8] })
/* 22866*/     0, // EndSwitchType
/* 22867*/    /*Scope*/ 44, /*->22912*/
/* 22868*/     OPC_CheckPredicate, 12, // Predicate_vtx_id1_load
/* 22870*/     OPC_SwitchType /*2 cases */, 18, MVT::v2i32,// ->22891
/* 22873*/      OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 22875*/      OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22878*/      OPC_EmitMergeInputChains1_0,
/* 22879*/      OPC_EmitInteger, MVT::i8, 1, 
/* 22882*/      OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_64_cm), 0|OPFL_Chain|OPFL_MemRefs,
                    MVT::v2i32, 3/*#Ops*/, 2, 3, 4, 
                // Src: (ld:{ *:[v2i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_load>><<P:Predicate_vtx_id1_load>> - Complexity = 13
                // Dst: (VTX_READ_64_cm:{ *:[v2i32] } MEMxi:{ *:[i32] }:$src_gpr, 1:{ *:[i8] })
/* 22891*/     /*SwitchType*/ 18, MVT::v4i32,// ->22911
/* 22893*/      OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 22895*/      OPC_CheckComplexPat, /*CP*/1, /*#*/1, // SelectADDRVTX_READ:$src_gpr #2 #3
/* 22898*/      OPC_EmitMergeInputChains1_0,
/* 22899*/      OPC_EmitInteger, MVT::i8, 1, 
/* 22902*/      OPC_MorphNodeTo1, TARGET_VAL(R600::VTX_READ_128_cm), 0|OPFL_Chain|OPFL_MemRefs,
                    MVT::v4i32, 3/*#Ops*/, 2, 3, 4, 
                // Src: (ld:{ *:[v4i32] } ADDRVTX_READ:{ *:[i32] }:$src_gpr)<<P:Predicate_unindexedload>><<P:Predicate_load>><<P:Predicate_vtx_id1_load>> - Complexity = 13
                // Dst: (VTX_READ_128_cm:{ *:[v4i32] } MEMxi:{ *:[i32] }:$src_gpr, 1:{ *:[i8] })
/* 22911*/     0, // EndSwitchType
/* 22912*/    0, /*End of Scope*/
/* 22913*/   0, /*End of Scope*/
/* 22914*/  0, /*End of Scope*/
/* 22915*/ /*SwitchOpcode*/ 74|128,2/*330*/, TARGET_VAL(ISD::STORE),// ->23249
/* 22919*/  OPC_RecordMemRef,
/* 22920*/  OPC_RecordNode, // #0 = 'st' chained node
/* 22921*/  OPC_RecordChild1, // #1 = $val
/* 22922*/  OPC_Scope, 119|128,1/*247*/, /*->23172*/ // 3 children in Scope
/* 22925*/   OPC_CheckChild1Type, MVT::i32,
/* 22927*/   OPC_RecordChild2, // #2 = $addr
/* 22928*/   OPC_CheckPredicate, 17, // Predicate_unindexedstore
/* 22930*/   OPC_Scope, 20, /*->22952*/ // 3 children in Scope
/* 22932*/    OPC_CheckPredicate, 18, // Predicate_store_private
/* 22934*/    OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 22936*/    OPC_CheckComplexPat, /*CP*/0, /*#*/2, // SelectADDRIndirect:$addr #3 #4
/* 22939*/    OPC_EmitMergeInputChains1_0,
/* 22940*/    OPC_EmitInteger, MVT::i32, 0, 
/* 22943*/    OPC_MorphNodeTo0, TARGET_VAL(R600::R600_RegisterStore), 0|OPFL_Chain|OPFL_MemRefs,
                  4/*#Ops*/, 1, 3, 4, 5, 
              // Src: (st i32:{ *:[i32] }:$val, ADDRIndirect:{ *:[iPTR] }:$addr)<<P:Predicate_unindexedstore>><<P:Predicate_store_private>> - Complexity = 13
              // Dst: (R600_RegisterStore i32:{ *:[i32] }:$val, FRAMEri:{ *:[iPTR] }:$addr, 0:{ *:[i32] })
/* 22952*/   /*Scope*/ 58, /*->23011*/
/* 22953*/    OPC_CheckPredicate, 19, // Predicate_store_local
/* 22955*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 22957*/    OPC_EmitMergeInputChains1_0,
/* 22958*/    OPC_EmitInteger, MVT::i32, 0, 
/* 22961*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 22973*/    OPC_EmitInteger, MVT::i32, 0, 
/* 22976*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 22988*/    OPC_EmitInteger, MVT::i32, 1, 
/* 22991*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 22994*/    OPC_EmitInteger, MVT::i32, 0, 
/* 22997*/    OPC_MorphNodeTo0, TARGET_VAL(R600::LDS_WRITE), 0|OPFL_Chain|OPFL_MemRefs,
                  9/*#Ops*/, 2, 3, 4, 1, 5, 6, 7, 8, 9, 
              // Src: (st R600_Reg32:{ *:[i32] }:$src1, R600_Reg32:{ *:[iPTR] }:$src0)<<P:Predicate_unindexedstore>><<P:Predicate_store_local>> - Complexity = 4
              // Dst: (LDS_WRITE R600_Reg32:{ *:[iPTR] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 23011*/   /*Scope*/ 30|128,1/*158*/, /*->23171*/
/* 23013*/    OPC_CheckChild2Type, MVT::i32,
/* 23015*/    OPC_Scope, 122, /*->23139*/ // 2 children in Scope
/* 23017*/     OPC_CheckPredicate, 20, // Predicate_truncstore
/* 23019*/     OPC_Scope, 58, /*->23079*/ // 2 children in Scope
/* 23021*/      OPC_CheckPredicate, 15, // Predicate_truncstorei8_local
/* 23023*/      OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 23025*/      OPC_EmitMergeInputChains1_0,
/* 23026*/      OPC_EmitInteger, MVT::i32, 0, 
/* 23029*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23041*/      OPC_EmitInteger, MVT::i32, 0, 
/* 23044*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23056*/      OPC_EmitInteger, MVT::i32, 1, 
/* 23059*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 23062*/      OPC_EmitInteger, MVT::i32, 0, 
/* 23065*/      OPC_MorphNodeTo0, TARGET_VAL(R600::LDS_BYTE_WRITE), 0|OPFL_Chain|OPFL_MemRefs,
                    9/*#Ops*/, 2, 3, 4, 1, 5, 6, 7, 8, 9, 
                // Src: (st i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src0)<<P:Predicate_unindexedstore>><<P:Predicate_truncstore>><<P:Predicate_truncstorei8_local>> - Complexity = 4
                // Dst: (LDS_BYTE_WRITE i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 23079*/     /*Scope*/ 58, /*->23138*/
/* 23080*/      OPC_CheckPredicate, 16, // Predicate_truncstorei16_local
/* 23082*/      OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 23084*/      OPC_EmitMergeInputChains1_0,
/* 23085*/      OPC_EmitInteger, MVT::i32, 0, 
/* 23088*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23100*/      OPC_EmitInteger, MVT::i32, 0, 
/* 23103*/      OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23115*/      OPC_EmitInteger, MVT::i32, 1, 
/* 23118*/      OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 23121*/      OPC_EmitInteger, MVT::i32, 0, 
/* 23124*/      OPC_MorphNodeTo0, TARGET_VAL(R600::LDS_SHORT_WRITE), 0|OPFL_Chain|OPFL_MemRefs,
                    9/*#Ops*/, 2, 3, 4, 1, 5, 6, 7, 8, 9, 
                // Src: (st i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src0)<<P:Predicate_unindexedstore>><<P:Predicate_truncstore>><<P:Predicate_truncstorei16_local>> - Complexity = 4
                // Dst: (LDS_SHORT_WRITE i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 23138*/     0, /*End of Scope*/
/* 23139*/    /*Scope*/ 30, /*->23170*/
/* 23140*/     OPC_CheckPredicate, 21, // Predicate_store_global
/* 23142*/     OPC_Scope, 10, /*->23154*/ // 2 children in Scope
/* 23144*/      OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 23146*/      OPC_EmitMergeInputChains1_0,
/* 23147*/      OPC_MorphNodeTo0, TARGET_VAL(R600::RAT_STORE_DWORD32), 0|OPFL_Chain|OPFL_MemRefs,
                    2/*#Ops*/, 1, 2, 
                // Src: (st i32:{ *:[i32] }:$rw_gpr, i32:{ *:[i32] }:$index_gpr)<<P:Predicate_unindexedstore>><<P:Predicate_store_global>> - Complexity = 4
                // Dst: (RAT_STORE_DWORD32 i32:{ *:[i32] }:$rw_gpr, i32:{ *:[i32] }:$index_gpr)
/* 23154*/     /*Scope*/ 14, /*->23169*/
/* 23155*/      OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 23157*/      OPC_EmitMergeInputChains1_0,
/* 23158*/      OPC_EmitInteger, MVT::i32, 0, 
/* 23161*/      OPC_MorphNodeTo0, TARGET_VAL(R600::RAT_WRITE_CACHELESS_32_eg), 0|OPFL_Chain|OPFL_MemRefs,
                    3/*#Ops*/, 1, 2, 3, 
                // Src: (st i32:{ *:[i32] }:$rw_gpr, i32:{ *:[i32] }:$index_gpr)<<P:Predicate_unindexedstore>><<P:Predicate_store_global>> - Complexity = 4
                // Dst: (RAT_WRITE_CACHELESS_32_eg i32:{ *:[i32] }:$rw_gpr, i32:{ *:[i32] }:$index_gpr)
/* 23169*/     0, /*End of Scope*/
/* 23170*/    0, /*End of Scope*/
/* 23171*/   0, /*End of Scope*/
/* 23172*/  /*Scope*/ 37, /*->23210*/
/* 23173*/   OPC_CheckChild1Type, MVT::v2i32,
/* 23175*/   OPC_RecordChild2, // #2 = $index_gpr
/* 23176*/   OPC_CheckChild2Type, MVT::i32,
/* 23178*/   OPC_CheckPredicate, 17, // Predicate_unindexedstore
/* 23180*/   OPC_CheckPredicate, 21, // Predicate_store_global
/* 23182*/   OPC_Scope, 10, /*->23194*/ // 2 children in Scope
/* 23184*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 23186*/    OPC_EmitMergeInputChains1_0,
/* 23187*/    OPC_MorphNodeTo0, TARGET_VAL(R600::RAT_STORE_DWORD64), 0|OPFL_Chain|OPFL_MemRefs,
                  2/*#Ops*/, 1, 2, 
              // Src: (st v2i32:{ *:[v2i32] }:$rw_gpr, i32:{ *:[i32] }:$index_gpr)<<P:Predicate_unindexedstore>><<P:Predicate_store_global>> - Complexity = 4
              // Dst: (RAT_STORE_DWORD64 v2i32:{ *:[v2i32] }:$rw_gpr, i32:{ *:[i32] }:$index_gpr)
/* 23194*/   /*Scope*/ 14, /*->23209*/
/* 23195*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 23197*/    OPC_EmitMergeInputChains1_0,
/* 23198*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23201*/    OPC_MorphNodeTo0, TARGET_VAL(R600::RAT_WRITE_CACHELESS_64_eg), 0|OPFL_Chain|OPFL_MemRefs,
                  3/*#Ops*/, 1, 2, 3, 
              // Src: (st v2i32:{ *:[v2i32] }:$rw_gpr, i32:{ *:[i32] }:$index_gpr)<<P:Predicate_unindexedstore>><<P:Predicate_store_global>> - Complexity = 4
              // Dst: (RAT_WRITE_CACHELESS_64_eg v2i32:{ *:[v2i32] }:$rw_gpr, i32:{ *:[i32] }:$index_gpr)
/* 23209*/   0, /*End of Scope*/
/* 23210*/  /*Scope*/ 37, /*->23248*/
/* 23211*/   OPC_CheckChild1Type, MVT::v4i32,
/* 23213*/   OPC_RecordChild2, // #2 = $index_gpr
/* 23214*/   OPC_CheckChild2Type, MVT::i32,
/* 23216*/   OPC_CheckPredicate, 17, // Predicate_unindexedstore
/* 23218*/   OPC_CheckPredicate, 21, // Predicate_store_global
/* 23220*/   OPC_Scope, 10, /*->23232*/ // 2 children in Scope
/* 23222*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 23224*/    OPC_EmitMergeInputChains1_0,
/* 23225*/    OPC_MorphNodeTo0, TARGET_VAL(R600::RAT_STORE_DWORD128), 0|OPFL_Chain|OPFL_MemRefs,
                  2/*#Ops*/, 1, 2, 
              // Src: (st v4i32:{ *:[v4i32] }:$rw_gpr, i32:{ *:[i32] }:$index_gpr)<<P:Predicate_unindexedstore>><<P:Predicate_store_global>> - Complexity = 4
              // Dst: (RAT_STORE_DWORD128 v4i32:{ *:[v4i32] }:$rw_gpr, i32:{ *:[i32] }:$index_gpr)
/* 23232*/   /*Scope*/ 14, /*->23247*/
/* 23233*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 23235*/    OPC_EmitMergeInputChains1_0,
/* 23236*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23239*/    OPC_MorphNodeTo0, TARGET_VAL(R600::RAT_WRITE_CACHELESS_128_eg), 0|OPFL_Chain|OPFL_MemRefs,
                  3/*#Ops*/, 1, 2, 3, 
              // Src: (st v4i32:{ *:[v4i32] }:$rw_gpr, i32:{ *:[i32] }:$index_gpr)<<P:Predicate_unindexedstore>><<P:Predicate_store_global>> - Complexity = 4
              // Dst: (RAT_WRITE_CACHELESS_128_eg v4i32:{ *:[v4i32] }:$rw_gpr, i32:{ *:[i32] }:$index_gpr)
/* 23247*/   0, /*End of Scope*/
/* 23248*/  0, /*End of Scope*/
/* 23249*/ /*SwitchOpcode*/ 90|128,7/*986*/, TARGET_VAL(ISD::ADD),// ->24239
/* 23253*/  OPC_Scope, 90|128,1/*218*/, /*->23474*/ // 4 children in Scope
/* 23256*/   OPC_MoveChild0,
/* 23257*/   OPC_CheckOpcode, TARGET_VAL(ISD::INTRINSIC_WO_CHAIN),
/* 23260*/   OPC_Scope, 105, /*->23367*/ // 2 children in Scope
/* 23262*/    OPC_CheckChild0Integer, 64|128,13/*1728*/, 
/* 23265*/    OPC_RecordChild1, // #0 = $src0
/* 23266*/    OPC_RecordChild2, // #1 = $src1
/* 23267*/    OPC_MoveParent,
/* 23268*/    OPC_RecordChild1, // #2 = $src2
/* 23269*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 23271*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23274*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23277*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23280*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23283*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23295*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23298*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23301*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23313*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23316*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23319*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23331*/    OPC_EmitInteger, MVT::i32, 1, 
/* 23334*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 23337*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23340*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23343*/    OPC_MorphNodeTo1, TARGET_VAL(R600::MULADD_UINT24_eg), 0,
                  MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
              // Src: (add:{ *:[i32] } (intrinsic_wo_chain:{ *:[i32] } 1728:{ *:[iPTR] }, i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1), i32:{ *:[i32] }:$src2) - Complexity = 13
              // Dst: (MULADD_UINT24_eg:{ *:[i32] } ?:{ *:[i32] }:$src0, ?:{ *:[i32] }:$src1, ?:{ *:[i32] }:$src2)
/* 23367*/   /*Scope*/ 105, /*->23473*/
/* 23368*/    OPC_CheckChild0Integer, 63|128,13/*1727*/, 
/* 23371*/    OPC_RecordChild1, // #0 = $src0
/* 23372*/    OPC_RecordChild2, // #1 = $src1
/* 23373*/    OPC_MoveParent,
/* 23374*/    OPC_RecordChild1, // #2 = $src2
/* 23375*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 23377*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23380*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23383*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23386*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23389*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23401*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23404*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23407*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23419*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23422*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23425*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23437*/    OPC_EmitInteger, MVT::i32, 1, 
/* 23440*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 23443*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23446*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23449*/    OPC_MorphNodeTo1, TARGET_VAL(R600::MULADD_INT24_cm), 0,
                  MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
              // Src: (add:{ *:[i32] } (intrinsic_wo_chain:{ *:[i32] } 1727:{ *:[iPTR] }, i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1), i32:{ *:[i32] }:$src2) - Complexity = 13
              // Dst: (MULADD_INT24_cm:{ *:[i32] } ?:{ *:[i32] }:$src0, ?:{ *:[i32] }:$src1, ?:{ *:[i32] }:$src2)
/* 23473*/   0, /*End of Scope*/
/* 23474*/  /*Scope*/ 89|128,1/*217*/, /*->23693*/
/* 23476*/   OPC_RecordChild0, // #0 = $src2
/* 23477*/   OPC_MoveChild1,
/* 23478*/   OPC_CheckOpcode, TARGET_VAL(ISD::INTRINSIC_WO_CHAIN),
/* 23481*/   OPC_Scope, 104, /*->23587*/ // 2 children in Scope
/* 23483*/    OPC_CheckChild0Integer, 64|128,13/*1728*/, 
/* 23486*/    OPC_RecordChild1, // #1 = $src0
/* 23487*/    OPC_RecordChild2, // #2 = $src1
/* 23488*/    OPC_MoveParent,
/* 23489*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 23491*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23494*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23497*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23500*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23503*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23515*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23518*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23521*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23533*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23536*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23539*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23551*/    OPC_EmitInteger, MVT::i32, 1, 
/* 23554*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 23557*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23560*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23563*/    OPC_MorphNodeTo1, TARGET_VAL(R600::MULADD_UINT24_eg), 0,
                  MVT::i32, 18/*#Ops*/, 3, 4, 1, 5, 6, 7, 2, 8, 9, 10, 0, 11, 12, 13, 14, 15, 16, 17, 
              // Src: (add:{ *:[i32] } i32:{ *:[i32] }:$src2, (intrinsic_wo_chain:{ *:[i32] } 1728:{ *:[iPTR] }, i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)) - Complexity = 13
              // Dst: (MULADD_UINT24_eg:{ *:[i32] } ?:{ *:[i32] }:$src0, ?:{ *:[i32] }:$src1, ?:{ *:[i32] }:$src2)
/* 23587*/   /*Scope*/ 104, /*->23692*/
/* 23588*/    OPC_CheckChild0Integer, 63|128,13/*1727*/, 
/* 23591*/    OPC_RecordChild1, // #1 = $src0
/* 23592*/    OPC_RecordChild2, // #2 = $src1
/* 23593*/    OPC_MoveParent,
/* 23594*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 23596*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23599*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23602*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23605*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23608*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23620*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23623*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23626*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23638*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23641*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23644*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23656*/    OPC_EmitInteger, MVT::i32, 1, 
/* 23659*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 23662*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23665*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23668*/    OPC_MorphNodeTo1, TARGET_VAL(R600::MULADD_INT24_cm), 0,
                  MVT::i32, 18/*#Ops*/, 3, 4, 1, 5, 6, 7, 2, 8, 9, 10, 0, 11, 12, 13, 14, 15, 16, 17, 
              // Src: (add:{ *:[i32] } i32:{ *:[i32] }:$src2, (intrinsic_wo_chain:{ *:[i32] } 1727:{ *:[iPTR] }, i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)) - Complexity = 13
              // Dst: (MULADD_INT24_cm:{ *:[i32] } ?:{ *:[i32] }:$src0, ?:{ *:[i32] }:$src1, ?:{ *:[i32] }:$src2)
/* 23692*/   0, /*End of Scope*/
/* 23693*/  /*Scope*/ 89|128,1/*217*/, /*->23912*/
/* 23695*/   OPC_MoveChild0,
/* 23696*/   OPC_SwitchOpcode /*2 cases */, 104, TARGET_VAL(AMDGPUISD::MUL_U24),// ->23804
/* 23700*/    OPC_RecordChild0, // #0 = $src0
/* 23701*/    OPC_RecordChild1, // #1 = $src1
/* 23702*/    OPC_MoveParent,
/* 23703*/    OPC_RecordChild1, // #2 = $src2
/* 23704*/    OPC_CheckType, MVT::i32,
/* 23706*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 23708*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23711*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23714*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23717*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23720*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23732*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23735*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23738*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23750*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23753*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23756*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23768*/    OPC_EmitInteger, MVT::i32, 1, 
/* 23771*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 23774*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23777*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23780*/    OPC_MorphNodeTo1, TARGET_VAL(R600::MULADD_UINT24_eg), 0,
                  MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
              // Src: (add:{ *:[i32] } (AMDGPUmul_u24_impl:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1), i32:{ *:[i32] }:$src2) - Complexity = 8
              // Dst: (MULADD_UINT24_eg:{ *:[i32] } ?:{ *:[i32] }:$src0, ?:{ *:[i32] }:$src1, ?:{ *:[i32] }:$src2)
/* 23804*/   /*SwitchOpcode*/ 104, TARGET_VAL(AMDGPUISD::MUL_I24),// ->23911
/* 23807*/    OPC_RecordChild0, // #0 = $src0
/* 23808*/    OPC_RecordChild1, // #1 = $src1
/* 23809*/    OPC_MoveParent,
/* 23810*/    OPC_RecordChild1, // #2 = $src2
/* 23811*/    OPC_CheckType, MVT::i32,
/* 23813*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 23815*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23818*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23821*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23824*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23827*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23839*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23842*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23845*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23857*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23860*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23863*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23875*/    OPC_EmitInteger, MVT::i32, 1, 
/* 23878*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 23881*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23884*/    OPC_EmitInteger, MVT::i32, 0, 
/* 23887*/    OPC_MorphNodeTo1, TARGET_VAL(R600::MULADD_INT24_cm), 0,
                  MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
              // Src: (add:{ *:[i32] } (AMDGPUmul_i24_impl:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1), i32:{ *:[i32] }:$src2) - Complexity = 8
              // Dst: (MULADD_INT24_cm:{ *:[i32] } ?:{ *:[i32] }:$src0, ?:{ *:[i32] }:$src1, ?:{ *:[i32] }:$src2)
/* 23911*/   0, // EndSwitchOpcode
/* 23912*/  /*Scope*/ 68|128,2/*324*/, /*->24238*/
/* 23914*/   OPC_RecordChild0, // #0 = $src2
/* 23915*/   OPC_Scope, 87|128,1/*215*/, /*->24133*/ // 2 children in Scope
/* 23918*/    OPC_MoveChild1,
/* 23919*/    OPC_SwitchOpcode /*2 cases */, 103, TARGET_VAL(AMDGPUISD::MUL_U24),// ->24026
/* 23923*/     OPC_RecordChild0, // #1 = $src0
/* 23924*/     OPC_RecordChild1, // #2 = $src1
/* 23925*/     OPC_MoveParent,
/* 23926*/     OPC_CheckType, MVT::i32,
/* 23928*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 23930*/     OPC_EmitInteger, MVT::i32, 0, 
/* 23933*/     OPC_EmitInteger, MVT::i32, 0, 
/* 23936*/     OPC_EmitInteger, MVT::i32, 0, 
/* 23939*/     OPC_EmitInteger, MVT::i32, 0, 
/* 23942*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23954*/     OPC_EmitInteger, MVT::i32, 0, 
/* 23957*/     OPC_EmitInteger, MVT::i32, 0, 
/* 23960*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23972*/     OPC_EmitInteger, MVT::i32, 0, 
/* 23975*/     OPC_EmitInteger, MVT::i32, 0, 
/* 23978*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 23990*/     OPC_EmitInteger, MVT::i32, 1, 
/* 23993*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 23996*/     OPC_EmitInteger, MVT::i32, 0, 
/* 23999*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24002*/     OPC_MorphNodeTo1, TARGET_VAL(R600::MULADD_UINT24_eg), 0,
                   MVT::i32, 18/*#Ops*/, 3, 4, 1, 5, 6, 7, 2, 8, 9, 10, 0, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (add:{ *:[i32] } i32:{ *:[i32] }:$src2, (AMDGPUmul_u24_impl:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)) - Complexity = 8
               // Dst: (MULADD_UINT24_eg:{ *:[i32] } ?:{ *:[i32] }:$src0, ?:{ *:[i32] }:$src1, ?:{ *:[i32] }:$src2)
/* 24026*/    /*SwitchOpcode*/ 103, TARGET_VAL(AMDGPUISD::MUL_I24),// ->24132
/* 24029*/     OPC_RecordChild0, // #1 = $src0
/* 24030*/     OPC_RecordChild1, // #2 = $src1
/* 24031*/     OPC_MoveParent,
/* 24032*/     OPC_CheckType, MVT::i32,
/* 24034*/     OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 24036*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24039*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24042*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24045*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24048*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24060*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24063*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24066*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24078*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24081*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24084*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24096*/     OPC_EmitInteger, MVT::i32, 1, 
/* 24099*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 24102*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24105*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24108*/     OPC_MorphNodeTo1, TARGET_VAL(R600::MULADD_INT24_cm), 0,
                   MVT::i32, 18/*#Ops*/, 3, 4, 1, 5, 6, 7, 2, 8, 9, 10, 0, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (add:{ *:[i32] } i32:{ *:[i32] }:$src2, (AMDGPUmul_i24_impl:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)) - Complexity = 8
               // Dst: (MULADD_INT24_cm:{ *:[i32] } ?:{ *:[i32] }:$src0, ?:{ *:[i32] }:$src1, ?:{ *:[i32] }:$src2)
/* 24132*/    0, // EndSwitchOpcode
/* 24133*/   /*Scope*/ 103, /*->24237*/
/* 24134*/    OPC_RecordChild1, // #1 = $src1
/* 24135*/    OPC_CheckType, MVT::i32,
/* 24137*/    OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 24139*/    OPC_EmitInteger, MVT::i32, 0, 
/* 24142*/    OPC_EmitInteger, MVT::i32, 0, 
/* 24145*/    OPC_EmitInteger, MVT::i32, 1, 
/* 24148*/    OPC_EmitInteger, MVT::i32, 0, 
/* 24151*/    OPC_EmitInteger, MVT::i32, 0, 
/* 24154*/    OPC_EmitInteger, MVT::i32, 0, 
/* 24157*/    OPC_EmitInteger, MVT::i32, 0, 
/* 24160*/    OPC_EmitInteger, MVT::i32, 0, 
/* 24163*/    OPC_EmitInteger, MVT::i32, 0, 
/* 24166*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24178*/    OPC_EmitInteger, MVT::i32, 0, 
/* 24181*/    OPC_EmitInteger, MVT::i32, 0, 
/* 24184*/    OPC_EmitInteger, MVT::i32, 0, 
/* 24187*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24199*/    OPC_EmitInteger, MVT::i32, 1, 
/* 24202*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 24205*/    OPC_EmitInteger, MVT::i32, 0, 
/* 24208*/    OPC_EmitInteger, MVT::i32, 0, 
/* 24211*/    OPC_MorphNodeTo1, TARGET_VAL(R600::ADD_INT), 0,
                  MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
              // Src: (add:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
              // Dst: (ADD_INT:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 24237*/   0, /*End of Scope*/
/* 24238*/  0, /*End of Scope*/
/* 24239*/ /*SwitchOpcode*/ 113|128,26/*3441*/, TARGET_VAL(ISD::XOR),// ->27684
/* 24243*/  OPC_Scope, 66|128,3/*450*/, /*->24696*/ // 5 children in Scope
/* 24246*/   OPC_RecordChild0, // #0 = $z
/* 24247*/   OPC_MoveChild1,
/* 24248*/   OPC_CheckOpcode, TARGET_VAL(ISD::AND),
/* 24251*/   OPC_Scope, 91|128,1/*219*/, /*->24473*/ // 2 children in Scope
/* 24254*/    OPC_RecordChild0, // #1 = $x
/* 24255*/    OPC_MoveChild1,
/* 24256*/    OPC_CheckOpcode, TARGET_VAL(ISD::XOR),
/* 24259*/    OPC_Scope, 105, /*->24366*/ // 2 children in Scope
/* 24261*/     OPC_RecordChild0, // #2 = $y
/* 24262*/     OPC_CheckChild1Same, 0,
/* 24264*/     OPC_MoveParent,
/* 24265*/     OPC_MoveParent,
/* 24266*/     OPC_CheckType, MVT::i32,
/* 24268*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 24270*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24273*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24276*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24279*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24282*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24294*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24297*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24300*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24312*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24315*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24318*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24330*/     OPC_EmitInteger, MVT::i32, 1, 
/* 24333*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 24336*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24339*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24342*/     OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 3, 4, 1, 5, 6, 7, 2, 8, 9, 10, 0, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (xor:{ *:[i32] } i32:{ *:[i32] }:$z, (and:{ *:[i32] } i32:{ *:[i32] }:$x, (xor:{ *:[i32] } i32:{ *:[i32] }:$y, i32:{ *:[i32] }:$z))) - Complexity = 9
               // Dst: (BFI_INT_eg:{ *:[i32] } ?:{ *:[i32] }:$x, ?:{ *:[i32] }:$y, ?:{ *:[i32] }:$z)
/* 24366*/    /*Scope*/ 105, /*->24472*/
/* 24367*/     OPC_CheckChild0Same, 0,
/* 24369*/     OPC_RecordChild1, // #2 = $y
/* 24370*/     OPC_MoveParent,
/* 24371*/     OPC_MoveParent,
/* 24372*/     OPC_CheckType, MVT::i32,
/* 24374*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 24376*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24379*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24382*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24385*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24388*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24400*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24403*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24406*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24418*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24421*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24424*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24436*/     OPC_EmitInteger, MVT::i32, 1, 
/* 24439*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 24442*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24445*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24448*/     OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 3, 4, 1, 5, 6, 7, 2, 8, 9, 10, 0, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (xor:{ *:[i32] } i32:{ *:[i32] }:$z, (and:{ *:[i32] } i32:{ *:[i32] }:$x, (xor:{ *:[i32] } i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$y))) - Complexity = 9
               // Dst: (BFI_INT_eg:{ *:[i32] } ?:{ *:[i32] }:$x, ?:{ *:[i32] }:$y, ?:{ *:[i32] }:$z)
/* 24472*/    0, /*End of Scope*/
/* 24473*/   /*Scope*/ 92|128,1/*220*/, /*->24695*/
/* 24475*/    OPC_MoveChild0,
/* 24476*/    OPC_CheckOpcode, TARGET_VAL(ISD::XOR),
/* 24479*/    OPC_Scope, 106, /*->24587*/ // 2 children in Scope
/* 24481*/     OPC_RecordChild0, // #1 = $y
/* 24482*/     OPC_CheckChild1Same, 0,
/* 24484*/     OPC_MoveParent,
/* 24485*/     OPC_RecordChild1, // #2 = $x
/* 24486*/     OPC_MoveParent,
/* 24487*/     OPC_CheckType, MVT::i32,
/* 24489*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 24491*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24494*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24497*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24500*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24503*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24515*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24518*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24521*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24533*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24536*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24539*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24551*/     OPC_EmitInteger, MVT::i32, 1, 
/* 24554*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 24557*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24560*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24563*/     OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 3, 4, 2, 5, 6, 7, 1, 8, 9, 10, 0, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (xor:{ *:[i32] } i32:{ *:[i32] }:$z, (and:{ *:[i32] } (xor:{ *:[i32] } i32:{ *:[i32] }:$y, i32:{ *:[i32] }:$z), i32:{ *:[i32] }:$x)) - Complexity = 9
               // Dst: (BFI_INT_eg:{ *:[i32] } ?:{ *:[i32] }:$x, ?:{ *:[i32] }:$y, ?:{ *:[i32] }:$z)
/* 24587*/    /*Scope*/ 106, /*->24694*/
/* 24588*/     OPC_CheckChild0Same, 0,
/* 24590*/     OPC_RecordChild1, // #1 = $y
/* 24591*/     OPC_MoveParent,
/* 24592*/     OPC_RecordChild1, // #2 = $x
/* 24593*/     OPC_MoveParent,
/* 24594*/     OPC_CheckType, MVT::i32,
/* 24596*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 24598*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24601*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24604*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24607*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24610*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24622*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24625*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24628*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24640*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24643*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24646*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24658*/     OPC_EmitInteger, MVT::i32, 1, 
/* 24661*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 24664*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24667*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24670*/     OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 3, 4, 2, 5, 6, 7, 1, 8, 9, 10, 0, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (xor:{ *:[i32] } i32:{ *:[i32] }:$z, (and:{ *:[i32] } (xor:{ *:[i32] } i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$y), i32:{ *:[i32] }:$x)) - Complexity = 9
               // Dst: (BFI_INT_eg:{ *:[i32] } ?:{ *:[i32] }:$x, ?:{ *:[i32] }:$y, ?:{ *:[i32] }:$z)
/* 24694*/    0, /*End of Scope*/
/* 24695*/   0, /*End of Scope*/
/* 24696*/  /*Scope*/ 56|128,3/*440*/, /*->25138*/
/* 24698*/   OPC_MoveChild0,
/* 24699*/   OPC_CheckOpcode, TARGET_VAL(ISD::AND),
/* 24702*/   OPC_Scope, 87|128,1/*215*/, /*->24920*/ // 2 children in Scope
/* 24705*/    OPC_RecordChild0, // #0 = $x
/* 24706*/    OPC_MoveChild1,
/* 24707*/    OPC_CheckOpcode, TARGET_VAL(ISD::XOR),
/* 24710*/    OPC_RecordChild0, // #1 = $y
/* 24711*/    OPC_RecordChild1, // #2 = $z
/* 24712*/    OPC_MoveParent,
/* 24713*/    OPC_MoveParent,
/* 24714*/    OPC_CheckType, MVT::i32,
/* 24716*/    OPC_Scope, 100, /*->24818*/ // 2 children in Scope
/* 24718*/     OPC_CheckChild1Same, 2,
/* 24720*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 24722*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24725*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24728*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24731*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24734*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24746*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24749*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24752*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24764*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24767*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24770*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24782*/     OPC_EmitInteger, MVT::i32, 1, 
/* 24785*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 24788*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24791*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24794*/     OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (xor:{ *:[i32] } (and:{ *:[i32] } i32:{ *:[i32] }:$x, (xor:{ *:[i32] } i32:{ *:[i32] }:$y, i32:{ *:[i32] }:$z)), i32:{ *:[i32] }:$z) - Complexity = 9
               // Dst: (BFI_INT_eg:{ *:[i32] } ?:{ *:[i32] }:$x, ?:{ *:[i32] }:$y, ?:{ *:[i32] }:$z)
/* 24818*/    /*Scope*/ 100, /*->24919*/
/* 24819*/     OPC_CheckChild1Same, 1,
/* 24821*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 24823*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24826*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24829*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24832*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24835*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24847*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24850*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24853*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24865*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24868*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24871*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24883*/     OPC_EmitInteger, MVT::i32, 1, 
/* 24886*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 24889*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24892*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24895*/     OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 2, 8, 9, 10, 1, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (xor:{ *:[i32] } (and:{ *:[i32] } i32:{ *:[i32] }:$x, (xor:{ *:[i32] } i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$y)), i32:{ *:[i32] }:$z) - Complexity = 9
               // Dst: (BFI_INT_eg:{ *:[i32] } ?:{ *:[i32] }:$x, ?:{ *:[i32] }:$y, ?:{ *:[i32] }:$z)
/* 24919*/    0, /*End of Scope*/
/* 24920*/   /*Scope*/ 87|128,1/*215*/, /*->25137*/
/* 24922*/    OPC_MoveChild0,
/* 24923*/    OPC_CheckOpcode, TARGET_VAL(ISD::XOR),
/* 24926*/    OPC_RecordChild0, // #0 = $y
/* 24927*/    OPC_RecordChild1, // #1 = $z
/* 24928*/    OPC_MoveParent,
/* 24929*/    OPC_RecordChild1, // #2 = $x
/* 24930*/    OPC_MoveParent,
/* 24931*/    OPC_CheckType, MVT::i32,
/* 24933*/    OPC_Scope, 100, /*->25035*/ // 2 children in Scope
/* 24935*/     OPC_CheckChild1Same, 1,
/* 24937*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 24939*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24942*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24945*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24948*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24951*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24963*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24966*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24969*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24981*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24984*/     OPC_EmitInteger, MVT::i32, 0, 
/* 24987*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 24999*/     OPC_EmitInteger, MVT::i32, 1, 
/* 25002*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 25005*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25008*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25011*/     OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 3, 4, 2, 5, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (xor:{ *:[i32] } (and:{ *:[i32] } (xor:{ *:[i32] } i32:{ *:[i32] }:$y, i32:{ *:[i32] }:$z), i32:{ *:[i32] }:$x), i32:{ *:[i32] }:$z) - Complexity = 9
               // Dst: (BFI_INT_eg:{ *:[i32] } ?:{ *:[i32] }:$x, ?:{ *:[i32] }:$y, ?:{ *:[i32] }:$z)
/* 25035*/    /*Scope*/ 100, /*->25136*/
/* 25036*/     OPC_CheckChild1Same, 0,
/* 25038*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 25040*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25043*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25046*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25049*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25052*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 25064*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25067*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25070*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 25082*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25085*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25088*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 25100*/     OPC_EmitInteger, MVT::i32, 1, 
/* 25103*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 25106*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25109*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25112*/     OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 3, 4, 2, 5, 6, 7, 1, 8, 9, 10, 0, 11, 12, 13, 14, 15, 16, 17, 
               // Src: (xor:{ *:[i32] } (and:{ *:[i32] } (xor:{ *:[i32] } i32:{ *:[i32] }:$z, i32:{ *:[i32] }:$y), i32:{ *:[i32] }:$x), i32:{ *:[i32] }:$z) - Complexity = 9
               // Dst: (BFI_INT_eg:{ *:[i32] } ?:{ *:[i32] }:$x, ?:{ *:[i32] }:$y, ?:{ *:[i32] }:$z)
/* 25136*/    0, /*End of Scope*/
/* 25137*/   0, /*End of Scope*/
/* 25138*/  /*Scope*/ 30|128,9/*1182*/, /*->26322*/
/* 25140*/   OPC_RecordChild0, // #0 = $z
/* 25141*/   OPC_MoveChild1,
/* 25142*/   OPC_CheckOpcode, TARGET_VAL(ISD::AND),
/* 25145*/   OPC_Scope, 73|128,4/*585*/, /*->25733*/ // 2 children in Scope
/* 25148*/    OPC_RecordChild0, // #1 = $x
/* 25149*/    OPC_MoveChild1,
/* 25150*/    OPC_CheckOpcode, TARGET_VAL(ISD::XOR),
/* 25153*/    OPC_Scope, 31|128,2/*287*/, /*->25443*/ // 2 children in Scope
/* 25156*/     OPC_RecordChild0, // #2 = $y
/* 25157*/     OPC_CheckChild1Same, 0,
/* 25159*/     OPC_MoveParent,
/* 25160*/     OPC_MoveParent,
/* 25161*/     OPC_CheckType, MVT::i64,
/* 25163*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 25165*/     OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 25168*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25171*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25174*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 25177*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 6,  // Results = #7
/* 25185*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25188*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25191*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 25203*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 25206*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 11,  // Results = #12
/* 25214*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25217*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25220*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 25232*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 25235*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 16,  // Results = #17
/* 25243*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25246*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25249*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 25261*/     OPC_EmitInteger, MVT::i32, 1, 
/* 25264*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 25267*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25270*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25273*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24,  // Results = #25
/* 25297*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 25300*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25303*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25306*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 25309*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 29,  // Results = #30
/* 25317*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25320*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25323*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 25335*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 25338*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 34,  // Results = #35
/* 25346*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25349*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25352*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 25364*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 25367*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 39,  // Results = #40
/* 25375*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25378*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25381*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 25393*/     OPC_EmitInteger, MVT::i32, 1, 
/* 25396*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 25399*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25402*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25405*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 27, 28, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47,  // Results = #48
/* 25429*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 25432*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                   MVT::i64, 5/*#Ops*/, 3, 25, 26, 48, 49, 
               // Src: (xor:{ *:[i64] } i64:{ *:[i64] }:$z, (and:{ *:[i64] } i64:{ *:[i64] }:$x, (xor:{ *:[i64] } i64:{ *:[i64] }:$y, i64:{ *:[i64] }:$z))) - Complexity = 9
               // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 25443*/    /*Scope*/ 31|128,2/*287*/, /*->25732*/
/* 25445*/     OPC_CheckChild0Same, 0,
/* 25447*/     OPC_RecordChild1, // #2 = $y
/* 25448*/     OPC_MoveParent,
/* 25449*/     OPC_MoveParent,
/* 25450*/     OPC_CheckType, MVT::i64,
/* 25452*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 25454*/     OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 25457*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25460*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25463*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 25466*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 6,  // Results = #7
/* 25474*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25477*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25480*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 25492*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 25495*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 11,  // Results = #12
/* 25503*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25506*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25509*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 25521*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 25524*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 16,  // Results = #17
/* 25532*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25535*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25538*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 25550*/     OPC_EmitInteger, MVT::i32, 1, 
/* 25553*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 25556*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25559*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25562*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24,  // Results = #25
/* 25586*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 25589*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25592*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25595*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 25598*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 29,  // Results = #30
/* 25606*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25609*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25612*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 25624*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 25627*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 34,  // Results = #35
/* 25635*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25638*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25641*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 25653*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 25656*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 39,  // Results = #40
/* 25664*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25667*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25670*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 25682*/     OPC_EmitInteger, MVT::i32, 1, 
/* 25685*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 25688*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25691*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25694*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 27, 28, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47,  // Results = #48
/* 25718*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 25721*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                   MVT::i64, 5/*#Ops*/, 3, 25, 26, 48, 49, 
               // Src: (xor:{ *:[i64] } i64:{ *:[i64] }:$z, (and:{ *:[i64] } i64:{ *:[i64] }:$x, (xor:{ *:[i64] } i64:{ *:[i64] }:$z, i64:{ *:[i64] }:$y))) - Complexity = 9
               // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 25732*/    0, /*End of Scope*/
/* 25733*/   /*Scope*/ 74|128,4/*586*/, /*->26321*/
/* 25735*/    OPC_MoveChild0,
/* 25736*/    OPC_CheckOpcode, TARGET_VAL(ISD::XOR),
/* 25739*/    OPC_Scope, 32|128,2/*288*/, /*->26030*/ // 2 children in Scope
/* 25742*/     OPC_RecordChild0, // #1 = $y
/* 25743*/     OPC_CheckChild1Same, 0,
/* 25745*/     OPC_MoveParent,
/* 25746*/     OPC_RecordChild1, // #2 = $x
/* 25747*/     OPC_MoveParent,
/* 25748*/     OPC_CheckType, MVT::i64,
/* 25750*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 25752*/     OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 25755*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25758*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25761*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 25764*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 6,  // Results = #7
/* 25772*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25775*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25778*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 25790*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 25793*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 11,  // Results = #12
/* 25801*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25804*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25807*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 25819*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 25822*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 16,  // Results = #17
/* 25830*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25833*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25836*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 25848*/     OPC_EmitInteger, MVT::i32, 1, 
/* 25851*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 25854*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25857*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25860*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24,  // Results = #25
/* 25884*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 25887*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25890*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25893*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 25896*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 29,  // Results = #30
/* 25904*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25907*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25910*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 25922*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 25925*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 34,  // Results = #35
/* 25933*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25936*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25939*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 25951*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 25954*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 39,  // Results = #40
/* 25962*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25965*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25968*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 25980*/     OPC_EmitInteger, MVT::i32, 1, 
/* 25983*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 25986*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25989*/     OPC_EmitInteger, MVT::i32, 0, 
/* 25992*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 27, 28, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47,  // Results = #48
/* 26016*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 26019*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                   MVT::i64, 5/*#Ops*/, 3, 25, 26, 48, 49, 
               // Src: (xor:{ *:[i64] } i64:{ *:[i64] }:$z, (and:{ *:[i64] } (xor:{ *:[i64] } i64:{ *:[i64] }:$y, i64:{ *:[i64] }:$z), i64:{ *:[i64] }:$x)) - Complexity = 9
               // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 26030*/    /*Scope*/ 32|128,2/*288*/, /*->26320*/
/* 26032*/     OPC_CheckChild0Same, 0,
/* 26034*/     OPC_RecordChild1, // #1 = $y
/* 26035*/     OPC_MoveParent,
/* 26036*/     OPC_RecordChild1, // #2 = $x
/* 26037*/     OPC_MoveParent,
/* 26038*/     OPC_CheckType, MVT::i64,
/* 26040*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 26042*/     OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 26045*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26048*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26051*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 26054*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 6,  // Results = #7
/* 26062*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26065*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26068*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 26080*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 26083*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 11,  // Results = #12
/* 26091*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26094*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26097*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 26109*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 26112*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 16,  // Results = #17
/* 26120*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26123*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26126*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 26138*/     OPC_EmitInteger, MVT::i32, 1, 
/* 26141*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 26144*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26147*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26150*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24,  // Results = #25
/* 26174*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 26177*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26180*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26183*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 26186*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 29,  // Results = #30
/* 26194*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26197*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26200*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 26212*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 26215*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 34,  // Results = #35
/* 26223*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26226*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26229*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 26241*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 26244*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 39,  // Results = #40
/* 26252*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26255*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26258*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 26270*/     OPC_EmitInteger, MVT::i32, 1, 
/* 26273*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 26276*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26279*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26282*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 27, 28, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47,  // Results = #48
/* 26306*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 26309*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                   MVT::i64, 5/*#Ops*/, 3, 25, 26, 48, 49, 
               // Src: (xor:{ *:[i64] } i64:{ *:[i64] }:$z, (and:{ *:[i64] } (xor:{ *:[i64] } i64:{ *:[i64] }:$z, i64:{ *:[i64] }:$y), i64:{ *:[i64] }:$x)) - Complexity = 9
               // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 26320*/    0, /*End of Scope*/
/* 26321*/   0, /*End of Scope*/
/* 26322*/  /*Scope*/ 20|128,9/*1172*/, /*->27496*/
/* 26324*/   OPC_MoveChild0,
/* 26325*/   OPC_CheckOpcode, TARGET_VAL(ISD::AND),
/* 26328*/   OPC_Scope, 69|128,4/*581*/, /*->26912*/ // 2 children in Scope
/* 26331*/    OPC_RecordChild0, // #0 = $x
/* 26332*/    OPC_MoveChild1,
/* 26333*/    OPC_CheckOpcode, TARGET_VAL(ISD::XOR),
/* 26336*/    OPC_RecordChild0, // #1 = $y
/* 26337*/    OPC_RecordChild1, // #2 = $z
/* 26338*/    OPC_MoveParent,
/* 26339*/    OPC_MoveParent,
/* 26340*/    OPC_CheckType, MVT::i64,
/* 26342*/    OPC_Scope, 26|128,2/*282*/, /*->26627*/ // 2 children in Scope
/* 26345*/     OPC_CheckChild1Same, 2,
/* 26347*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 26349*/     OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 26352*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26355*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26358*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 26361*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 6,  // Results = #7
/* 26369*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26372*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26375*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 26387*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 26390*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 11,  // Results = #12
/* 26398*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26401*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26404*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 26416*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 26419*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 16,  // Results = #17
/* 26427*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26430*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26433*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 26445*/     OPC_EmitInteger, MVT::i32, 1, 
/* 26448*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 26451*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26454*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26457*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24,  // Results = #25
/* 26481*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 26484*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26487*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26490*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 26493*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 29,  // Results = #30
/* 26501*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26504*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26507*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 26519*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 26522*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 34,  // Results = #35
/* 26530*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26533*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26536*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 26548*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 26551*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 39,  // Results = #40
/* 26559*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26562*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26565*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 26577*/     OPC_EmitInteger, MVT::i32, 1, 
/* 26580*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 26583*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26586*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26589*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 27, 28, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47,  // Results = #48
/* 26613*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 26616*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                   MVT::i64, 5/*#Ops*/, 3, 25, 26, 48, 49, 
               // Src: (xor:{ *:[i64] } (and:{ *:[i64] } i64:{ *:[i64] }:$x, (xor:{ *:[i64] } i64:{ *:[i64] }:$y, i64:{ *:[i64] }:$z)), i64:{ *:[i64] }:$z) - Complexity = 9
               // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 26627*/    /*Scope*/ 26|128,2/*282*/, /*->26911*/
/* 26629*/     OPC_CheckChild1Same, 1,
/* 26631*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 26633*/     OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 26636*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26639*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26642*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 26645*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 6,  // Results = #7
/* 26653*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26656*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26659*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 26671*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 26674*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 11,  // Results = #12
/* 26682*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26685*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26688*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 26700*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 26703*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 16,  // Results = #17
/* 26711*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26714*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26717*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 26729*/     OPC_EmitInteger, MVT::i32, 1, 
/* 26732*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 26735*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26738*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26741*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24,  // Results = #25
/* 26765*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 26768*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26771*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26774*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 26777*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 29,  // Results = #30
/* 26785*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26788*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26791*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 26803*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 26806*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 34,  // Results = #35
/* 26814*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26817*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26820*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 26832*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 26835*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 39,  // Results = #40
/* 26843*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26846*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26849*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 26861*/     OPC_EmitInteger, MVT::i32, 1, 
/* 26864*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 26867*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26870*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26873*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 27, 28, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47,  // Results = #48
/* 26897*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 26900*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                   MVT::i64, 5/*#Ops*/, 3, 25, 26, 48, 49, 
               // Src: (xor:{ *:[i64] } (and:{ *:[i64] } i64:{ *:[i64] }:$x, (xor:{ *:[i64] } i64:{ *:[i64] }:$z, i64:{ *:[i64] }:$y)), i64:{ *:[i64] }:$z) - Complexity = 9
               // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 26911*/    0, /*End of Scope*/
/* 26912*/   /*Scope*/ 69|128,4/*581*/, /*->27495*/
/* 26914*/    OPC_MoveChild0,
/* 26915*/    OPC_CheckOpcode, TARGET_VAL(ISD::XOR),
/* 26918*/    OPC_RecordChild0, // #0 = $y
/* 26919*/    OPC_RecordChild1, // #1 = $z
/* 26920*/    OPC_MoveParent,
/* 26921*/    OPC_RecordChild1, // #2 = $x
/* 26922*/    OPC_MoveParent,
/* 26923*/    OPC_CheckType, MVT::i64,
/* 26925*/    OPC_Scope, 26|128,2/*282*/, /*->27210*/ // 2 children in Scope
/* 26928*/     OPC_CheckChild1Same, 1,
/* 26930*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 26932*/     OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 26935*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26938*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26941*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 26944*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 6,  // Results = #7
/* 26952*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26955*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26958*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 26970*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 26973*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 11,  // Results = #12
/* 26981*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26984*/     OPC_EmitInteger, MVT::i32, 0, 
/* 26987*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 26999*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 27002*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 16,  // Results = #17
/* 27010*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27013*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27016*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 27028*/     OPC_EmitInteger, MVT::i32, 1, 
/* 27031*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 27034*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27037*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27040*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24,  // Results = #25
/* 27064*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 27067*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27070*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27073*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 27076*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 29,  // Results = #30
/* 27084*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27087*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27090*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 27102*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 27105*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 34,  // Results = #35
/* 27113*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27116*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27119*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 27131*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 27134*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 39,  // Results = #40
/* 27142*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27145*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27148*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 27160*/     OPC_EmitInteger, MVT::i32, 1, 
/* 27163*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 27166*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27169*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27172*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 27, 28, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47,  // Results = #48
/* 27196*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 27199*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                   MVT::i64, 5/*#Ops*/, 3, 25, 26, 48, 49, 
               // Src: (xor:{ *:[i64] } (and:{ *:[i64] } (xor:{ *:[i64] } i64:{ *:[i64] }:$y, i64:{ *:[i64] }:$z), i64:{ *:[i64] }:$x), i64:{ *:[i64] }:$z) - Complexity = 9
               // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 27210*/    /*Scope*/ 26|128,2/*282*/, /*->27494*/
/* 27212*/     OPC_CheckChild1Same, 0,
/* 27214*/     OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 27216*/     OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 27219*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27222*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27225*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 27228*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 6,  // Results = #7
/* 27236*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27239*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27242*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 27254*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 27257*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 11,  // Results = #12
/* 27265*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27268*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27271*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 27283*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 27286*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 16,  // Results = #17
/* 27294*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27297*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27300*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 27312*/     OPC_EmitInteger, MVT::i32, 1, 
/* 27315*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 27318*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27321*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27324*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24,  // Results = #25
/* 27348*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 27351*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27354*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27357*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 27360*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 2, 29,  // Results = #30
/* 27368*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27371*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27374*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 27386*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 27389*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 1, 34,  // Results = #35
/* 27397*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27400*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27403*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 27415*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 27418*/     OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 39,  // Results = #40
/* 27426*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27429*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27432*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 27444*/     OPC_EmitInteger, MVT::i32, 1, 
/* 27447*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 27450*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27453*/     OPC_EmitInteger, MVT::i32, 0, 
/* 27456*/     OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                   MVT::i32, 18/*#Ops*/, 27, 28, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47,  // Results = #48
/* 27480*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 27483*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                   MVT::i64, 5/*#Ops*/, 3, 25, 26, 48, 49, 
               // Src: (xor:{ *:[i64] } (and:{ *:[i64] } (xor:{ *:[i64] } i64:{ *:[i64] }:$z, i64:{ *:[i64] }:$y), i64:{ *:[i64] }:$x), i64:{ *:[i64] }:$z) - Complexity = 9
               // Dst: (REG_SEQUENCE:{ *:[i64] } R600_Reg64:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub0:{ *:[i32] })), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$x, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$y, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[i64] }:$z, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 27494*/    0, /*End of Scope*/
/* 27495*/   0, /*End of Scope*/
/* 27496*/  /*Scope*/ 57|128,1/*185*/, /*->27683*/
/* 27498*/   OPC_RecordChild0, // #0 = $src0
/* 27499*/   OPC_CheckType, MVT::i32,
/* 27501*/   OPC_Scope, 77, /*->27580*/ // 2 children in Scope
/* 27503*/    OPC_CheckChild1Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 27514*/    OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 27516*/    OPC_EmitInteger, MVT::i32, 1, 
/* 27519*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27522*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27525*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27528*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27531*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27534*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27537*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 27549*/    OPC_EmitInteger, MVT::i32, 1, 
/* 27552*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 27555*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27558*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27561*/    OPC_MorphNodeTo1, TARGET_VAL(R600::NOT_INT), 0,
                  MVT::i32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
              // Src: (xor:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, -1:{ *:[i32] }) - Complexity = 8
              // Dst: (NOT_INT:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0)
/* 27580*/   /*Scope*/ 101, /*->27682*/
/* 27581*/    OPC_RecordChild1, // #1 = $src1
/* 27582*/    OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 27584*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27587*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27590*/    OPC_EmitInteger, MVT::i32, 1, 
/* 27593*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27596*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27599*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27602*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27605*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27608*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27611*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 27623*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27626*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27629*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27632*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 27644*/    OPC_EmitInteger, MVT::i32, 1, 
/* 27647*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 27650*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27653*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27656*/    OPC_MorphNodeTo1, TARGET_VAL(R600::XOR_INT), 0,
                  MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
              // Src: (xor:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
              // Dst: (XOR_INT:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 27682*/   0, /*End of Scope*/
/* 27683*/  0, /*End of Scope*/
/* 27684*/ /*SwitchOpcode*/ 45, TARGET_VAL(AMDGPUISD::CONST_ADDRESS),// ->27732
/* 27687*/  OPC_RecordChild0, // #0 = $src
/* 27688*/  OPC_CheckChild0Type, MVT::i32,
/* 27690*/  OPC_Scope, 12, /*->27704*/ // 2 children in Scope
/* 27692*/   OPC_CheckType, MVT::i32,
/* 27694*/   OPC_CheckComplexPat, /*CP*/2, /*#*/0, // SelectGlobalValueConstantOffset:$src #1
/* 27697*/   OPC_MorphNodeTo1, TARGET_VAL(R600::CONST_COPY), 0|OPFL_Variadic1,
                 MVT::i32, 1/*#Ops*/, 1, 
             // Src: (CONST_ADDRESS:{ *:[i32] } ADDRGA_CONST_OFFSET:{ *:[i32] }:$src) - Complexity = 9
             // Dst: (CONST_COPY:{ *:[i32] } ADDRGA_CONST_OFFSET:{ *:[i32] }:$src)
/* 27704*/  /*Scope*/ 26, /*->27731*/
/* 27705*/   OPC_RecordChild1, // #1 = $buffer_id
/* 27706*/   OPC_MoveChild1,
/* 27707*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 27710*/   OPC_CheckType, MVT::i32,
/* 27712*/   OPC_MoveParent,
/* 27713*/   OPC_CheckType, MVT::v4i32,
/* 27715*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 27717*/   OPC_CheckComplexPat, /*CP*/3, /*#*/0, // SelectGlobalValueVariableOffset:$ptr #2 #3
/* 27720*/   OPC_EmitConvertToTarget, 1,
/* 27722*/   OPC_MorphNodeTo1, TARGET_VAL(R600::TEX_VTX_CONSTBUF), 0|OPFL_Variadic2,
                 MVT::v4i32, 3/*#Ops*/, 2, 3, 4, 
             // Src: (CONST_ADDRESS:{ *:[v4i32] } ADDRGA_VAR_OFFSET:{ *:[i32] }:$ptr, (imm:{ *:[i32] }):$buffer_id) - Complexity = 15
             // Dst: (TEX_VTX_CONSTBUF:{ *:[v4i32] } ADDRGA_VAR_OFFSET:{ *:[i32] }:$ptr, (imm:{ *:[i32] }):$buffer_id)
/* 27731*/  0, /*End of Scope*/
/* 27732*/ /*SwitchOpcode*/ 125|128,1/*253*/, TARGET_VAL(ISD::ATOMIC_LOAD_ADD),// ->27989
/* 27736*/  OPC_RecordMemRef,
/* 27737*/  OPC_RecordNode, // #0 = 'atomic_load_add' chained node
/* 27738*/  OPC_RecordChild1, // #1 = $ptr
/* 27739*/  OPC_CheckChild1Type, MVT::i32,
/* 27741*/  OPC_CheckType, MVT::i32,
/* 27743*/  OPC_Scope, 63, /*->27808*/ // 3 children in Scope
/* 27745*/   OPC_CheckChild2Integer, 1, 
/* 27747*/   OPC_CheckPredicate, 22, // Predicate_atomic_load_add_global_noret_32
/* 27749*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 27751*/   OPC_EmitMergeInputChains1_0,
/* 27752*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::IMPLICIT_DEF), 0,
                 MVT::v4i32, 0/*#Ops*/,  // Results = #2
/* 27758*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 27770*/   OPC_EmitNode1, TARGET_VAL(R600::MOV_IMM_I32), 0,
                 MVT::i32, 1/*#Ops*/, 3,  // Results = #4
/* 27777*/   OPC_EmitInteger, MVT::i32, R600::sub0,
/* 27780*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                 MVT::v4i32, 3/*#Ops*/, 2, 4, 5,  // Results = #6
/* 27789*/   OPC_EmitNode1, TARGET_VAL(R600::RAT_ATOMIC_INC_UINT_NORET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::v4i32, 2/*#Ops*/, 6, 1,  // Results = #7
/* 27797*/   OPC_EmitInteger, MVT::i32, R600::sub1,
/* 27800*/   OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0|OPFL_Chain,
                 MVT::i32, 2/*#Ops*/, 7, 8, 
             // Src: (atomic_load_add:{ *:[i32] } i32:{ *:[i32] }:$ptr, 1:{ *:[i32] })<<P:Predicate_atomic_load_add_global_noret_32>> - Complexity = 9
             // Dst: (EXTRACT_SUBREG:{ *:[i32] } (RAT_ATOMIC_INC_UINT_NORET:{ *:[v4i32] } (INSERT_SUBREG:{ *:[v4i32] } (IMPLICIT_DEF:{ *:[v4i32] }), (MOV_IMM_I32:{ *:[i32] } -1:{ *:[i32] }), sub0:{ *:[i32] }), ?:{ *:[i32] }:$ptr), sub1:{ *:[i32] })
/* 27808*/  /*Scope*/ 72, /*->27881*/
/* 27809*/   OPC_CheckChild2Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 27820*/   OPC_CheckPredicate, 22, // Predicate_atomic_load_add_global_noret_32
/* 27822*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 27824*/   OPC_EmitMergeInputChains1_0,
/* 27825*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::IMPLICIT_DEF), 0,
                 MVT::v4i32, 0/*#Ops*/,  // Results = #2
/* 27831*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 27843*/   OPC_EmitNode1, TARGET_VAL(R600::MOV_IMM_I32), 0,
                 MVT::i32, 1/*#Ops*/, 3,  // Results = #4
/* 27850*/   OPC_EmitInteger, MVT::i32, R600::sub0,
/* 27853*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                 MVT::v4i32, 3/*#Ops*/, 2, 4, 5,  // Results = #6
/* 27862*/   OPC_EmitNode1, TARGET_VAL(R600::RAT_ATOMIC_DEC_UINT_NORET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::v4i32, 2/*#Ops*/, 6, 1,  // Results = #7
/* 27870*/   OPC_EmitInteger, MVT::i32, R600::sub1,
/* 27873*/   OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0|OPFL_Chain,
                 MVT::i32, 2/*#Ops*/, 7, 8, 
             // Src: (atomic_load_add:{ *:[i32] } i32:{ *:[i32] }:$ptr, -1:{ *:[i32] })<<P:Predicate_atomic_load_add_global_noret_32>> - Complexity = 9
             // Dst: (EXTRACT_SUBREG:{ *:[i32] } (RAT_ATOMIC_DEC_UINT_NORET:{ *:[v4i32] } (INSERT_SUBREG:{ *:[v4i32] } (IMPLICIT_DEF:{ *:[v4i32] }), (MOV_IMM_I32:{ *:[i32] } -1:{ *:[i32] }), sub0:{ *:[i32] }), ?:{ *:[i32] }:$ptr), sub1:{ *:[i32] })
/* 27881*/  /*Scope*/ 106, /*->27988*/
/* 27882*/   OPC_RecordChild2, // #2 = $data
/* 27883*/   OPC_Scope, 42, /*->27927*/ // 2 children in Scope
/* 27885*/    OPC_CheckPredicate, 22, // Predicate_atomic_load_add_global_noret_32
/* 27887*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 27889*/    OPC_EmitMergeInputChains1_0,
/* 27890*/    OPC_EmitNode1, TARGET_VAL(TargetOpcode::IMPLICIT_DEF), 0,
                  MVT::v4i32, 0/*#Ops*/,  // Results = #3
/* 27896*/    OPC_EmitInteger, MVT::i32, R600::sub0,
/* 27899*/    OPC_EmitNode1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                  MVT::v4i32, 3/*#Ops*/, 3, 2, 4,  // Results = #5
/* 27908*/    OPC_EmitNode1, TARGET_VAL(R600::RAT_ATOMIC_ADD_NORET), 0|OPFL_Chain|OPFL_MemRefs,
                  MVT::v4i32, 2/*#Ops*/, 5, 1,  // Results = #6
/* 27916*/    OPC_EmitInteger, MVT::i32, R600::sub1,
/* 27919*/    OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0|OPFL_Chain,
                  MVT::i32, 2/*#Ops*/, 6, 7, 
              // Src: (atomic_load_add:{ *:[i32] } i32:{ *:[i32] }:$ptr, i32:{ *:[i32] }:$data)<<P:Predicate_atomic_load_add_global_noret_32>> - Complexity = 4
              // Dst: (EXTRACT_SUBREG:{ *:[i32] } (RAT_ATOMIC_ADD_NORET:{ *:[v4i32] } (INSERT_SUBREG:{ *:[v4i32] } (IMPLICIT_DEF:{ *:[v4i32] }), ?:{ *:[i32] }:$data, sub0:{ *:[i32] }), ?:{ *:[i32] }:$ptr), sub1:{ *:[i32] })
/* 27927*/   /*Scope*/ 59, /*->27987*/
/* 27928*/    OPC_CheckPredicate, 23, // Predicate_atomic_load_add_local_32
/* 27930*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 27932*/    OPC_EmitMergeInputChains1_0,
/* 27933*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27936*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 27948*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27951*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 27963*/    OPC_EmitInteger, MVT::i32, 1, 
/* 27966*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 27969*/    OPC_EmitInteger, MVT::i32, 0, 
/* 27972*/    OPC_MorphNodeTo1, TARGET_VAL(R600::LDS_ADD_RET), 0|OPFL_Chain|OPFL_MemRefs,
                  MVT::i32, 9/*#Ops*/, 1, 3, 4, 2, 5, 6, 7, 8, 9, 
              // Src: (atomic_load_add:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)<<P:Predicate_atomic_load_add_local_32>> - Complexity = 4
              // Dst: (LDS_ADD_RET:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 27987*/   0, /*End of Scope*/
/* 27988*/  0, /*End of Scope*/
/* 27989*/ /*SwitchOpcode*/ 125|128,1/*253*/, TARGET_VAL(ISD::ATOMIC_LOAD_SUB),// ->28246
/* 27993*/  OPC_RecordMemRef,
/* 27994*/  OPC_RecordNode, // #0 = 'atomic_load_sub' chained node
/* 27995*/  OPC_RecordChild1, // #1 = $ptr
/* 27996*/  OPC_CheckChild1Type, MVT::i32,
/* 27998*/  OPC_CheckType, MVT::i32,
/* 28000*/  OPC_Scope, 72, /*->28074*/ // 3 children in Scope
/* 28002*/   OPC_CheckChild2Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 28013*/   OPC_CheckPredicate, 22, // Predicate_atomic_load_sub_global_noret_32
/* 28015*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28017*/   OPC_EmitMergeInputChains1_0,
/* 28018*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::IMPLICIT_DEF), 0,
                 MVT::v4i32, 0/*#Ops*/,  // Results = #2
/* 28024*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 28036*/   OPC_EmitNode1, TARGET_VAL(R600::MOV_IMM_I32), 0,
                 MVT::i32, 1/*#Ops*/, 3,  // Results = #4
/* 28043*/   OPC_EmitInteger, MVT::i32, R600::sub0,
/* 28046*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                 MVT::v4i32, 3/*#Ops*/, 2, 4, 5,  // Results = #6
/* 28055*/   OPC_EmitNode1, TARGET_VAL(R600::RAT_ATOMIC_INC_UINT_NORET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::v4i32, 2/*#Ops*/, 6, 1,  // Results = #7
/* 28063*/   OPC_EmitInteger, MVT::i32, R600::sub1,
/* 28066*/   OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0|OPFL_Chain,
                 MVT::i32, 2/*#Ops*/, 7, 8, 
             // Src: (atomic_load_sub:{ *:[i32] } i32:{ *:[i32] }:$ptr, -1:{ *:[i32] })<<P:Predicate_atomic_load_sub_global_noret_32>> - Complexity = 9
             // Dst: (EXTRACT_SUBREG:{ *:[i32] } (RAT_ATOMIC_INC_UINT_NORET:{ *:[v4i32] } (INSERT_SUBREG:{ *:[v4i32] } (IMPLICIT_DEF:{ *:[v4i32] }), (MOV_IMM_I32:{ *:[i32] } -1:{ *:[i32] }), sub0:{ *:[i32] }), ?:{ *:[i32] }:$ptr), sub1:{ *:[i32] })
/* 28074*/  /*Scope*/ 63, /*->28138*/
/* 28075*/   OPC_CheckChild2Integer, 1, 
/* 28077*/   OPC_CheckPredicate, 22, // Predicate_atomic_load_sub_global_noret_32
/* 28079*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28081*/   OPC_EmitMergeInputChains1_0,
/* 28082*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::IMPLICIT_DEF), 0,
                 MVT::v4i32, 0/*#Ops*/,  // Results = #2
/* 28088*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 28100*/   OPC_EmitNode1, TARGET_VAL(R600::MOV_IMM_I32), 0,
                 MVT::i32, 1/*#Ops*/, 3,  // Results = #4
/* 28107*/   OPC_EmitInteger, MVT::i32, R600::sub0,
/* 28110*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                 MVT::v4i32, 3/*#Ops*/, 2, 4, 5,  // Results = #6
/* 28119*/   OPC_EmitNode1, TARGET_VAL(R600::RAT_ATOMIC_DEC_UINT_NORET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::v4i32, 2/*#Ops*/, 6, 1,  // Results = #7
/* 28127*/   OPC_EmitInteger, MVT::i32, R600::sub1,
/* 28130*/   OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0|OPFL_Chain,
                 MVT::i32, 2/*#Ops*/, 7, 8, 
             // Src: (atomic_load_sub:{ *:[i32] } i32:{ *:[i32] }:$ptr, 1:{ *:[i32] })<<P:Predicate_atomic_load_sub_global_noret_32>> - Complexity = 9
             // Dst: (EXTRACT_SUBREG:{ *:[i32] } (RAT_ATOMIC_DEC_UINT_NORET:{ *:[v4i32] } (INSERT_SUBREG:{ *:[v4i32] } (IMPLICIT_DEF:{ *:[v4i32] }), (MOV_IMM_I32:{ *:[i32] } -1:{ *:[i32] }), sub0:{ *:[i32] }), ?:{ *:[i32] }:$ptr), sub1:{ *:[i32] })
/* 28138*/  /*Scope*/ 106, /*->28245*/
/* 28139*/   OPC_RecordChild2, // #2 = $data
/* 28140*/   OPC_Scope, 42, /*->28184*/ // 2 children in Scope
/* 28142*/    OPC_CheckPredicate, 22, // Predicate_atomic_load_sub_global_noret_32
/* 28144*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28146*/    OPC_EmitMergeInputChains1_0,
/* 28147*/    OPC_EmitNode1, TARGET_VAL(TargetOpcode::IMPLICIT_DEF), 0,
                  MVT::v4i32, 0/*#Ops*/,  // Results = #3
/* 28153*/    OPC_EmitInteger, MVT::i32, R600::sub0,
/* 28156*/    OPC_EmitNode1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                  MVT::v4i32, 3/*#Ops*/, 3, 2, 4,  // Results = #5
/* 28165*/    OPC_EmitNode1, TARGET_VAL(R600::RAT_ATOMIC_SUB_NORET), 0|OPFL_Chain|OPFL_MemRefs,
                  MVT::v4i32, 2/*#Ops*/, 5, 1,  // Results = #6
/* 28173*/    OPC_EmitInteger, MVT::i32, R600::sub1,
/* 28176*/    OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0|OPFL_Chain,
                  MVT::i32, 2/*#Ops*/, 6, 7, 
              // Src: (atomic_load_sub:{ *:[i32] } i32:{ *:[i32] }:$ptr, i32:{ *:[i32] }:$data)<<P:Predicate_atomic_load_sub_global_noret_32>> - Complexity = 4
              // Dst: (EXTRACT_SUBREG:{ *:[i32] } (RAT_ATOMIC_SUB_NORET:{ *:[v4i32] } (INSERT_SUBREG:{ *:[v4i32] } (IMPLICIT_DEF:{ *:[v4i32] }), ?:{ *:[i32] }:$data, sub0:{ *:[i32] }), ?:{ *:[i32] }:$ptr), sub1:{ *:[i32] })
/* 28184*/   /*Scope*/ 59, /*->28244*/
/* 28185*/    OPC_CheckPredicate, 23, // Predicate_atomic_load_sub_local_32
/* 28187*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28189*/    OPC_EmitMergeInputChains1_0,
/* 28190*/    OPC_EmitInteger, MVT::i32, 0, 
/* 28193*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 28205*/    OPC_EmitInteger, MVT::i32, 0, 
/* 28208*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 28220*/    OPC_EmitInteger, MVT::i32, 1, 
/* 28223*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 28226*/    OPC_EmitInteger, MVT::i32, 0, 
/* 28229*/    OPC_MorphNodeTo1, TARGET_VAL(R600::LDS_SUB_RET), 0|OPFL_Chain|OPFL_MemRefs,
                  MVT::i32, 9/*#Ops*/, 1, 3, 4, 2, 5, 6, 7, 8, 9, 
              // Src: (atomic_load_sub:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)<<P:Predicate_atomic_load_sub_local_32>> - Complexity = 4
              // Dst: (LDS_SUB_RET:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 28244*/   0, /*End of Scope*/
/* 28245*/  0, /*End of Scope*/
/* 28246*/ /*SwitchOpcode*/ 25|128,2/*281*/, TARGET_VAL(ISD::EXTRACT_VECTOR_ELT),// ->28531
/* 28250*/  OPC_RecordChild0, // #0 = $src
/* 28251*/  OPC_SwitchType /*2 cases */, 8|128,1/*136*/, MVT::i32,// ->28391
/* 28255*/   OPC_Scope, 82, /*->28339*/ // 2 children in Scope
/* 28257*/    OPC_CheckChild0Type, MVT::v4i32,
/* 28259*/    OPC_Scope, 15, /*->28276*/ // 5 children in Scope
/* 28261*/     OPC_CheckChild1Integer, 0, 
/* 28263*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28265*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 28268*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 1, 
               // Src: (extractelt:{ *:[i32] } v4i32:{ *:[v4i32] }:$src, 0:{ *:[iPTR] }) - Complexity = 8
               // Dst: (EXTRACT_SUBREG:{ *:[i32] } ?:{ *:[v4i32] }:$src, sub0:{ *:[i32] })
/* 28276*/    /*Scope*/ 15, /*->28292*/
/* 28277*/     OPC_CheckChild1Integer, 1, 
/* 28279*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28281*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 28284*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 1, 
               // Src: (extractelt:{ *:[i32] } v4i32:{ *:[v4i32] }:$src, 1:{ *:[iPTR] }) - Complexity = 8
               // Dst: (EXTRACT_SUBREG:{ *:[i32] } ?:{ *:[v4i32] }:$src, sub1:{ *:[i32] })
/* 28292*/    /*Scope*/ 15, /*->28308*/
/* 28293*/     OPC_CheckChild1Integer, 2, 
/* 28295*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28297*/     OPC_EmitInteger, MVT::i32, R600::sub2,
/* 28300*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 1, 
               // Src: (extractelt:{ *:[i32] } v4i32:{ *:[v4i32] }:$src, 2:{ *:[iPTR] }) - Complexity = 8
               // Dst: (EXTRACT_SUBREG:{ *:[i32] } ?:{ *:[v4i32] }:$src, sub2:{ *:[i32] })
/* 28308*/    /*Scope*/ 15, /*->28324*/
/* 28309*/     OPC_CheckChild1Integer, 3, 
/* 28311*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28313*/     OPC_EmitInteger, MVT::i32, R600::sub3,
/* 28316*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 1, 
               // Src: (extractelt:{ *:[i32] } v4i32:{ *:[v4i32] }:$src, 3:{ *:[iPTR] }) - Complexity = 8
               // Dst: (EXTRACT_SUBREG:{ *:[i32] } ?:{ *:[v4i32] }:$src, sub3:{ *:[i32] })
/* 28324*/    /*Scope*/ 13, /*->28338*/
/* 28325*/     OPC_RecordChild1, // #1 = $index
/* 28326*/     OPC_CheckChild1Type, MVT::i32,
/* 28328*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28330*/     OPC_MorphNodeTo1, TARGET_VAL(R600::R600_EXTRACT_ELT_V4), 0,
                   MVT::i32, 2/*#Ops*/, 0, 1, 
               // Src: (extractelt:{ *:[i32] } v4i32:{ *:[v4i32] }:$vec, i32:{ *:[i32] }:$index) - Complexity = 3
               // Dst: (R600_EXTRACT_ELT_V4:{ *:[i32] } ?:{ *:[v4i32] }:$vec, ?:{ *:[i32] }:$index)
/* 28338*/    0, /*End of Scope*/
/* 28339*/   /*Scope*/ 50, /*->28390*/
/* 28340*/    OPC_CheckChild0Type, MVT::v2i32,
/* 28342*/    OPC_Scope, 15, /*->28359*/ // 3 children in Scope
/* 28344*/     OPC_CheckChild1Integer, 0, 
/* 28346*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28348*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 28351*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 1, 
               // Src: (extractelt:{ *:[i32] } v2i32:{ *:[v2i32] }:$src, 0:{ *:[iPTR] }) - Complexity = 8
               // Dst: (EXTRACT_SUBREG:{ *:[i32] } ?:{ *:[v2i32] }:$src, sub0:{ *:[i32] })
/* 28359*/    /*Scope*/ 15, /*->28375*/
/* 28360*/     OPC_CheckChild1Integer, 1, 
/* 28362*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28364*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 28367*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::i32, 2/*#Ops*/, 0, 1, 
               // Src: (extractelt:{ *:[i32] } v2i32:{ *:[v2i32] }:$src, 1:{ *:[iPTR] }) - Complexity = 8
               // Dst: (EXTRACT_SUBREG:{ *:[i32] } ?:{ *:[v2i32] }:$src, sub1:{ *:[i32] })
/* 28375*/    /*Scope*/ 13, /*->28389*/
/* 28376*/     OPC_RecordChild1, // #1 = $index
/* 28377*/     OPC_CheckChild1Type, MVT::i32,
/* 28379*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28381*/     OPC_MorphNodeTo1, TARGET_VAL(R600::R600_EXTRACT_ELT_V2), 0,
                   MVT::i32, 2/*#Ops*/, 0, 1, 
               // Src: (extractelt:{ *:[i32] } v2i32:{ *:[v2i32] }:$vec, i32:{ *:[i32] }:$index) - Complexity = 3
               // Dst: (R600_EXTRACT_ELT_V2:{ *:[i32] } ?:{ *:[v2i32] }:$vec, ?:{ *:[i32] }:$index)
/* 28389*/    0, /*End of Scope*/
/* 28390*/   0, /*End of Scope*/
/* 28391*/  /*SwitchType*/ 8|128,1/*136*/, MVT::f32,// ->28530
/* 28394*/   OPC_Scope, 82, /*->28478*/ // 2 children in Scope
/* 28396*/    OPC_CheckChild0Type, MVT::v4f32,
/* 28398*/    OPC_Scope, 15, /*->28415*/ // 5 children in Scope
/* 28400*/     OPC_CheckChild1Integer, 0, 
/* 28402*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28404*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 28407*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::f32, 2/*#Ops*/, 0, 1, 
               // Src: (extractelt:{ *:[f32] } v4f32:{ *:[v4f32] }:$src, 0:{ *:[iPTR] }) - Complexity = 8
               // Dst: (EXTRACT_SUBREG:{ *:[f32] } ?:{ *:[v4f32] }:$src, sub0:{ *:[i32] })
/* 28415*/    /*Scope*/ 15, /*->28431*/
/* 28416*/     OPC_CheckChild1Integer, 1, 
/* 28418*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28420*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 28423*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::f32, 2/*#Ops*/, 0, 1, 
               // Src: (extractelt:{ *:[f32] } v4f32:{ *:[v4f32] }:$src, 1:{ *:[iPTR] }) - Complexity = 8
               // Dst: (EXTRACT_SUBREG:{ *:[f32] } ?:{ *:[v4f32] }:$src, sub1:{ *:[i32] })
/* 28431*/    /*Scope*/ 15, /*->28447*/
/* 28432*/     OPC_CheckChild1Integer, 2, 
/* 28434*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28436*/     OPC_EmitInteger, MVT::i32, R600::sub2,
/* 28439*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::f32, 2/*#Ops*/, 0, 1, 
               // Src: (extractelt:{ *:[f32] } v4f32:{ *:[v4f32] }:$src, 2:{ *:[iPTR] }) - Complexity = 8
               // Dst: (EXTRACT_SUBREG:{ *:[f32] } ?:{ *:[v4f32] }:$src, sub2:{ *:[i32] })
/* 28447*/    /*Scope*/ 15, /*->28463*/
/* 28448*/     OPC_CheckChild1Integer, 3, 
/* 28450*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28452*/     OPC_EmitInteger, MVT::i32, R600::sub3,
/* 28455*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::f32, 2/*#Ops*/, 0, 1, 
               // Src: (extractelt:{ *:[f32] } v4f32:{ *:[v4f32] }:$src, 3:{ *:[iPTR] }) - Complexity = 8
               // Dst: (EXTRACT_SUBREG:{ *:[f32] } ?:{ *:[v4f32] }:$src, sub3:{ *:[i32] })
/* 28463*/    /*Scope*/ 13, /*->28477*/
/* 28464*/     OPC_RecordChild1, // #1 = $index
/* 28465*/     OPC_CheckChild1Type, MVT::i32,
/* 28467*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28469*/     OPC_MorphNodeTo1, TARGET_VAL(R600::R600_EXTRACT_ELT_V4), 0,
                   MVT::f32, 2/*#Ops*/, 0, 1, 
               // Src: (extractelt:{ *:[f32] } v4f32:{ *:[v4f32] }:$vec, i32:{ *:[i32] }:$index) - Complexity = 3
               // Dst: (R600_EXTRACT_ELT_V4:{ *:[f32] } ?:{ *:[v4f32] }:$vec, ?:{ *:[i32] }:$index)
/* 28477*/    0, /*End of Scope*/
/* 28478*/   /*Scope*/ 50, /*->28529*/
/* 28479*/    OPC_CheckChild0Type, MVT::v2f32,
/* 28481*/    OPC_Scope, 15, /*->28498*/ // 3 children in Scope
/* 28483*/     OPC_CheckChild1Integer, 0, 
/* 28485*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28487*/     OPC_EmitInteger, MVT::i32, R600::sub0,
/* 28490*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::f32, 2/*#Ops*/, 0, 1, 
               // Src: (extractelt:{ *:[f32] } v2f32:{ *:[v2f32] }:$src, 0:{ *:[iPTR] }) - Complexity = 8
               // Dst: (EXTRACT_SUBREG:{ *:[f32] } ?:{ *:[v2f32] }:$src, sub0:{ *:[i32] })
/* 28498*/    /*Scope*/ 15, /*->28514*/
/* 28499*/     OPC_CheckChild1Integer, 1, 
/* 28501*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28503*/     OPC_EmitInteger, MVT::i32, R600::sub1,
/* 28506*/     OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                   MVT::f32, 2/*#Ops*/, 0, 1, 
               // Src: (extractelt:{ *:[f32] } v2f32:{ *:[v2f32] }:$src, 1:{ *:[iPTR] }) - Complexity = 8
               // Dst: (EXTRACT_SUBREG:{ *:[f32] } ?:{ *:[v2f32] }:$src, sub1:{ *:[i32] })
/* 28514*/    /*Scope*/ 13, /*->28528*/
/* 28515*/     OPC_RecordChild1, // #1 = $index
/* 28516*/     OPC_CheckChild1Type, MVT::i32,
/* 28518*/     OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28520*/     OPC_MorphNodeTo1, TARGET_VAL(R600::R600_EXTRACT_ELT_V2), 0,
                   MVT::f32, 2/*#Ops*/, 0, 1, 
               // Src: (extractelt:{ *:[f32] } v2f32:{ *:[v2f32] }:$vec, i32:{ *:[i32] }:$index) - Complexity = 3
               // Dst: (R600_EXTRACT_ELT_V2:{ *:[f32] } ?:{ *:[v2f32] }:$vec, ?:{ *:[i32] }:$index)
/* 28528*/    0, /*End of Scope*/
/* 28529*/   0, /*End of Scope*/
/* 28530*/  0, // EndSwitchType
/* 28531*/ /*SwitchOpcode*/ 72|128,12/*1608*/, TARGET_VAL(ISD::INTRINSIC_WO_CHAIN),// ->30143
/* 28535*/  OPC_Scope, 108, /*->28645*/ // 10 children in Scope
/* 28537*/   OPC_CheckChild0Integer, 23|128,14/*1815*/, 
/* 28540*/   OPC_RecordChild1, // #0 = $src0
/* 28541*/   OPC_CheckChild1Type, MVT::i32,
/* 28543*/   OPC_RecordChild2, // #1 = $src1
/* 28544*/   OPC_RecordChild3, // #2 = $src2
/* 28545*/   OPC_CheckType, MVT::i32,
/* 28547*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28549*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28552*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28555*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28558*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28561*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 28573*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28576*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28579*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 28591*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28594*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28597*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 28609*/   OPC_EmitInteger, MVT::i32, 1, 
/* 28612*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 28615*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28618*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28621*/   OPC_MorphNodeTo1, TARGET_VAL(R600::BFE_UINT_eg), 0,
                 MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
             // Src: (intrinsic_wo_chain:{ *:[i32] } 1815:{ *:[iPTR] }, i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2) - Complexity = 8
             // Dst: (BFE_UINT_eg:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2)
/* 28645*/  /*Scope*/ 108, /*->28754*/
/* 28646*/   OPC_CheckChild0Integer, 119|128,13/*1783*/, 
/* 28649*/   OPC_RecordChild1, // #0 = $src0
/* 28650*/   OPC_CheckChild1Type, MVT::i32,
/* 28652*/   OPC_RecordChild2, // #1 = $src1
/* 28653*/   OPC_RecordChild3, // #2 = $src2
/* 28654*/   OPC_CheckType, MVT::i32,
/* 28656*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28658*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28661*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28664*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28667*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28670*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 28682*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28685*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28688*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 28700*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28703*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28706*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 28718*/   OPC_EmitInteger, MVT::i32, 1, 
/* 28721*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 28724*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28727*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28730*/   OPC_MorphNodeTo1, TARGET_VAL(R600::BFE_INT_eg), 0,
                 MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
             // Src: (intrinsic_wo_chain:{ *:[i32] } 1783:{ *:[iPTR] }, i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2) - Complexity = 8
             // Dst: (BFE_INT_eg:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2)
/* 28754*/  /*Scope*/ 105, /*->28860*/
/* 28755*/   OPC_CheckChild0Integer, 64|128,13/*1728*/, 
/* 28758*/   OPC_RecordChild1, // #0 = $src0
/* 28759*/   OPC_RecordChild2, // #1 = $src1
/* 28760*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 28762*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28765*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28768*/   OPC_EmitInteger, MVT::i32, 1, 
/* 28771*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28774*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28777*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28780*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28783*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28786*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28789*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 28801*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28804*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28807*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28810*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 28822*/   OPC_EmitInteger, MVT::i32, 1, 
/* 28825*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 28828*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28831*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28834*/   OPC_MorphNodeTo1, TARGET_VAL(R600::MUL_UINT24_eg), 0,
                 MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
             // Src: (intrinsic_wo_chain:{ *:[i32] } 1728:{ *:[iPTR] }, i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1) - Complexity = 8
             // Dst: (MUL_UINT24_eg:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 28860*/  /*Scope*/ 105, /*->28966*/
/* 28861*/   OPC_CheckChild0Integer, 63|128,13/*1727*/, 
/* 28864*/   OPC_RecordChild1, // #0 = $src0
/* 28865*/   OPC_RecordChild2, // #1 = $src1
/* 28866*/   OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 28868*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28871*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28874*/   OPC_EmitInteger, MVT::i32, 1, 
/* 28877*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28880*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28883*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28886*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28889*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28892*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28895*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 28907*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28910*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28913*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28916*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 28928*/   OPC_EmitInteger, MVT::i32, 1, 
/* 28931*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 28934*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28937*/   OPC_EmitInteger, MVT::i32, 0, 
/* 28940*/   OPC_MorphNodeTo1, TARGET_VAL(R600::MUL_INT24_cm), 0,
                 MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
             // Src: (intrinsic_wo_chain:{ *:[i32] } 1727:{ *:[iPTR] }, i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1) - Complexity = 8
             // Dst: (MUL_INT24_cm:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 28966*/  /*Scope*/ 46|128,3/*430*/, /*->29398*/
/* 28968*/   OPC_CheckChild0Integer, 90|128,13/*1754*/, 
/* 28971*/   OPC_Scope, 85|128,1/*213*/, /*->29187*/ // 2 children in Scope
/* 28974*/    OPC_MoveChild1,
/* 28975*/    OPC_CheckOpcode, TARGET_VAL(ISD::FSQRT),
/* 28978*/    OPC_RecordChild0, // #0 = $src
/* 28979*/    OPC_CheckType, MVT::f32,
/* 28981*/    OPC_MoveParent,
/* 28982*/    OPC_CheckType, MVT::f32,
/* 28984*/    OPC_Scope, 66, /*->29052*/ // 3 children in Scope
/* 28986*/     OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 28988*/     OPC_EmitInteger, MVT::i32, 1, 
/* 28991*/     OPC_EmitInteger, MVT::i32, 0, 
/* 28994*/     OPC_EmitInteger, MVT::i32, 0, 
/* 28997*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29000*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29003*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29006*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29009*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 29021*/     OPC_EmitInteger, MVT::i32, 1, 
/* 29024*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 29027*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29030*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29033*/     OPC_MorphNodeTo1, TARGET_VAL(R600::RECIPSQRT_IEEE_r600), 0,
                   MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
               // Src: (intrinsic_wo_chain:{ *:[f32] } 1754:{ *:[iPTR] }, (fsqrt:{ *:[f32] } f32:{ *:[f32] }:$src)) - Complexity = 11
               // Dst: (RECIPSQRT_IEEE_r600:{ *:[f32] } ?:{ *:[f32] }:$src)
/* 29052*/    /*Scope*/ 66, /*->29119*/
/* 29053*/     OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 29055*/     OPC_EmitInteger, MVT::i32, 1, 
/* 29058*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29061*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29064*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29067*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29070*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29073*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29076*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 29088*/     OPC_EmitInteger, MVT::i32, 1, 
/* 29091*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 29094*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29097*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29100*/     OPC_MorphNodeTo1, TARGET_VAL(R600::RECIPSQRT_IEEE_eg), 0,
                   MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
               // Src: (intrinsic_wo_chain:{ *:[f32] } 1754:{ *:[iPTR] }, (fsqrt:{ *:[f32] } f32:{ *:[f32] }:$src)) - Complexity = 11
               // Dst: (RECIPSQRT_IEEE_eg:{ *:[f32] } ?:{ *:[f32] }:$src)
/* 29119*/    /*Scope*/ 66, /*->29186*/
/* 29120*/     OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 29122*/     OPC_EmitInteger, MVT::i32, 1, 
/* 29125*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29128*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29131*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29134*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29137*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29140*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29143*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 29155*/     OPC_EmitInteger, MVT::i32, 1, 
/* 29158*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 29161*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29164*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29167*/     OPC_MorphNodeTo1, TARGET_VAL(R600::RECIPSQRT_IEEE_cm), 0,
                   MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
               // Src: (intrinsic_wo_chain:{ *:[f32] } 1754:{ *:[iPTR] }, (fsqrt:{ *:[f32] } f32:{ *:[f32] }:$src)) - Complexity = 11
               // Dst: (RECIPSQRT_IEEE_cm:{ *:[f32] } ?:{ *:[f32] }:$src)
/* 29186*/    0, /*End of Scope*/
/* 29187*/   /*Scope*/ 80|128,1/*208*/, /*->29397*/
/* 29189*/    OPC_RecordChild1, // #0 = $src0
/* 29190*/    OPC_CheckChild1Type, MVT::f32,
/* 29192*/    OPC_CheckType, MVT::f32,
/* 29194*/    OPC_Scope, 66, /*->29262*/ // 3 children in Scope
/* 29196*/     OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 29198*/     OPC_EmitInteger, MVT::i32, 1, 
/* 29201*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29204*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29207*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29210*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29213*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29216*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29219*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 29231*/     OPC_EmitInteger, MVT::i32, 1, 
/* 29234*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 29237*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29240*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29243*/     OPC_MorphNodeTo1, TARGET_VAL(R600::RECIP_IEEE_r600), 0,
                   MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
               // Src: (intrinsic_wo_chain:{ *:[f32] } 1754:{ *:[iPTR] }, f32:{ *:[f32] }:$src0) - Complexity = 8
               // Dst: (RECIP_IEEE_r600:{ *:[f32] } f32:{ *:[f32] }:$src0)
/* 29262*/    /*Scope*/ 66, /*->29329*/
/* 29263*/     OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 29265*/     OPC_EmitInteger, MVT::i32, 1, 
/* 29268*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29271*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29274*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29277*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29280*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29283*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29286*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 29298*/     OPC_EmitInteger, MVT::i32, 1, 
/* 29301*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 29304*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29307*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29310*/     OPC_MorphNodeTo1, TARGET_VAL(R600::RECIP_IEEE_eg), 0,
                   MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
               // Src: (intrinsic_wo_chain:{ *:[f32] } 1754:{ *:[iPTR] }, f32:{ *:[f32] }:$src0) - Complexity = 8
               // Dst: (RECIP_IEEE_eg:{ *:[f32] } f32:{ *:[f32] }:$src0)
/* 29329*/    /*Scope*/ 66, /*->29396*/
/* 29330*/     OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 29332*/     OPC_EmitInteger, MVT::i32, 1, 
/* 29335*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29338*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29341*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29344*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29347*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29350*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29353*/     OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 29365*/     OPC_EmitInteger, MVT::i32, 1, 
/* 29368*/     OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 29371*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29374*/     OPC_EmitInteger, MVT::i32, 0, 
/* 29377*/     OPC_MorphNodeTo1, TARGET_VAL(R600::RECIP_IEEE_cm), 0,
                   MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
               // Src: (intrinsic_wo_chain:{ *:[f32] } 1754:{ *:[iPTR] }, f32:{ *:[f32] }:$src0) - Complexity = 8
               // Dst: (RECIP_IEEE_cm:{ *:[f32] } f32:{ *:[f32] }:$src0)
/* 29396*/    0, /*End of Scope*/
/* 29397*/   0, /*End of Scope*/
/* 29398*/  /*Scope*/ 74, /*->29473*/
/* 29399*/   OPC_CheckChild0Integer, 55|128,9/*1207*/, 
/* 29402*/   OPC_RecordChild1, // #0 = $src0
/* 29403*/   OPC_CheckChild1Type, MVT::f32,
/* 29405*/   OPC_CheckType, MVT::f32,
/* 29407*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 29409*/   OPC_EmitInteger, MVT::i32, 1, 
/* 29412*/   OPC_EmitInteger, MVT::i32, 0, 
/* 29415*/   OPC_EmitInteger, MVT::i32, 0, 
/* 29418*/   OPC_EmitInteger, MVT::i32, 0, 
/* 29421*/   OPC_EmitInteger, MVT::i32, 0, 
/* 29424*/   OPC_EmitInteger, MVT::i32, 0, 
/* 29427*/   OPC_EmitInteger, MVT::i32, 0, 
/* 29430*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 29442*/   OPC_EmitInteger, MVT::i32, 1, 
/* 29445*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 29448*/   OPC_EmitInteger, MVT::i32, 0, 
/* 29451*/   OPC_EmitInteger, MVT::i32, 0, 
/* 29454*/   OPC_MorphNodeTo1, TARGET_VAL(R600::FRACT), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (intrinsic_wo_chain:{ *:[f32] } 1207:{ *:[iPTR] }, R600_Reg32:{ *:[f32] }:$src0) - Complexity = 8
             // Dst: (FRACT:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 29473*/  /*Scope*/ 86|128,1/*214*/, /*->29689*/
/* 29475*/   OPC_CheckChild0Integer, 52|128,9/*1204*/, 
/* 29478*/   OPC_RecordChild1, // #0 = $src0
/* 29479*/   OPC_CheckChild1Type, MVT::f32,
/* 29481*/   OPC_RecordChild2, // #1 = $src1
/* 29482*/   OPC_CheckChild2Type, MVT::f32,
/* 29484*/   OPC_RecordChild3, // #2 = $src2
/* 29485*/   OPC_CheckChild3Type, MVT::f32,
/* 29487*/   OPC_CheckType, MVT::f32,
/* 29489*/   OPC_Scope, 98, /*->29589*/ // 2 children in Scope
/* 29491*/    OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 29493*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29496*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29499*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29502*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29505*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 29517*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29520*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29523*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 29535*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29538*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29541*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 29553*/    OPC_EmitInteger, MVT::i32, 1, 
/* 29556*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 29559*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29562*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29565*/    OPC_MorphNodeTo1, TARGET_VAL(R600::MULADD_IEEE_r600), 0,
                  MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
              // Src: (intrinsic_wo_chain:{ *:[f32] } 1204:{ *:[iPTR] }, f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2) - Complexity = 8
              // Dst: (MULADD_IEEE_r600:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2)
/* 29589*/   /*Scope*/ 98, /*->29688*/
/* 29590*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 29592*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29595*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29598*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29601*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29604*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 29616*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29619*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29622*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 29634*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29637*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29640*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 29652*/    OPC_EmitInteger, MVT::i32, 1, 
/* 29655*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 29658*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29661*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29664*/    OPC_MorphNodeTo1, TARGET_VAL(R600::MULADD_IEEE_eg), 0,
                  MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
              // Src: (intrinsic_wo_chain:{ *:[f32] } 1204:{ *:[iPTR] }, f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2) - Complexity = 8
              // Dst: (MULADD_IEEE_eg:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2)
/* 29688*/   0, /*End of Scope*/
/* 29689*/  /*Scope*/ 83|128,1/*211*/, /*->29902*/
/* 29691*/   OPC_CheckChild0Integer, 96|128,13/*1760*/, 
/* 29694*/   OPC_RecordChild1, // #0 = $src0
/* 29695*/   OPC_CheckChild1Type, MVT::f32,
/* 29697*/   OPC_CheckType, MVT::f32,
/* 29699*/   OPC_Scope, 66, /*->29767*/ // 3 children in Scope
/* 29701*/    OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 29703*/    OPC_EmitInteger, MVT::i32, 1, 
/* 29706*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29709*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29712*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29715*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29718*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29721*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29724*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 29736*/    OPC_EmitInteger, MVT::i32, 1, 
/* 29739*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 29742*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29745*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29748*/    OPC_MorphNodeTo1, TARGET_VAL(R600::RECIPSQRT_CLAMPED_r600), 0,
                  MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
              // Src: (intrinsic_wo_chain:{ *:[f32] } 1760:{ *:[iPTR] }, R600_Reg32:{ *:[f32] }:$src0) - Complexity = 8
              // Dst: (RECIPSQRT_CLAMPED_r600:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 29767*/   /*Scope*/ 66, /*->29834*/
/* 29768*/    OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 29770*/    OPC_EmitInteger, MVT::i32, 1, 
/* 29773*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29776*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29779*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29782*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29785*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29788*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29791*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 29803*/    OPC_EmitInteger, MVT::i32, 1, 
/* 29806*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 29809*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29812*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29815*/    OPC_MorphNodeTo1, TARGET_VAL(R600::RECIPSQRT_CLAMPED_eg), 0,
                  MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
              // Src: (intrinsic_wo_chain:{ *:[f32] } 1760:{ *:[iPTR] }, R600_Reg32:{ *:[f32] }:$src0) - Complexity = 8
              // Dst: (RECIPSQRT_CLAMPED_eg:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 29834*/   /*Scope*/ 66, /*->29901*/
/* 29835*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 29837*/    OPC_EmitInteger, MVT::i32, 1, 
/* 29840*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29843*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29846*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29849*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29852*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29855*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29858*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 29870*/    OPC_EmitInteger, MVT::i32, 1, 
/* 29873*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 29876*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29879*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29882*/    OPC_MorphNodeTo1, TARGET_VAL(R600::RECIPSQRT_CLAMPED_cm), 0,
                  MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
              // Src: (intrinsic_wo_chain:{ *:[f32] } 1760:{ *:[iPTR] }, R600_Reg32:{ *:[f32] }:$src0) - Complexity = 8
              // Dst: (RECIPSQRT_CLAMPED_cm:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 29901*/   0, /*End of Scope*/
/* 29902*/  /*Scope*/ 83|128,1/*211*/, /*->30115*/
/* 29904*/   OPC_CheckChild0Integer, 95|128,13/*1759*/, 
/* 29907*/   OPC_RecordChild1, // #0 = $src0
/* 29908*/   OPC_CheckChild1Type, MVT::f32,
/* 29910*/   OPC_CheckType, MVT::f32,
/* 29912*/   OPC_Scope, 66, /*->29980*/ // 3 children in Scope
/* 29914*/    OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 29916*/    OPC_EmitInteger, MVT::i32, 1, 
/* 29919*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29922*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29925*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29928*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29931*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29934*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29937*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 29949*/    OPC_EmitInteger, MVT::i32, 1, 
/* 29952*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 29955*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29958*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29961*/    OPC_MorphNodeTo1, TARGET_VAL(R600::RECIPSQRT_IEEE_r600), 0,
                  MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
              // Src: (intrinsic_wo_chain:{ *:[f32] } 1759:{ *:[iPTR] }, R600_Reg32:{ *:[f32] }:$src0) - Complexity = 8
              // Dst: (RECIPSQRT_IEEE_r600:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 29980*/   /*Scope*/ 66, /*->30047*/
/* 29981*/    OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 29983*/    OPC_EmitInteger, MVT::i32, 1, 
/* 29986*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29989*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29992*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29995*/    OPC_EmitInteger, MVT::i32, 0, 
/* 29998*/    OPC_EmitInteger, MVT::i32, 0, 
/* 30001*/    OPC_EmitInteger, MVT::i32, 0, 
/* 30004*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 30016*/    OPC_EmitInteger, MVT::i32, 1, 
/* 30019*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 30022*/    OPC_EmitInteger, MVT::i32, 0, 
/* 30025*/    OPC_EmitInteger, MVT::i32, 0, 
/* 30028*/    OPC_MorphNodeTo1, TARGET_VAL(R600::RECIPSQRT_IEEE_eg), 0,
                  MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
              // Src: (intrinsic_wo_chain:{ *:[f32] } 1759:{ *:[iPTR] }, R600_Reg32:{ *:[f32] }:$src0) - Complexity = 8
              // Dst: (RECIPSQRT_IEEE_eg:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 30047*/   /*Scope*/ 66, /*->30114*/
/* 30048*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 30050*/    OPC_EmitInteger, MVT::i32, 1, 
/* 30053*/    OPC_EmitInteger, MVT::i32, 0, 
/* 30056*/    OPC_EmitInteger, MVT::i32, 0, 
/* 30059*/    OPC_EmitInteger, MVT::i32, 0, 
/* 30062*/    OPC_EmitInteger, MVT::i32, 0, 
/* 30065*/    OPC_EmitInteger, MVT::i32, 0, 
/* 30068*/    OPC_EmitInteger, MVT::i32, 0, 
/* 30071*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 30083*/    OPC_EmitInteger, MVT::i32, 1, 
/* 30086*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 30089*/    OPC_EmitInteger, MVT::i32, 0, 
/* 30092*/    OPC_EmitInteger, MVT::i32, 0, 
/* 30095*/    OPC_MorphNodeTo1, TARGET_VAL(R600::RECIPSQRT_IEEE_cm), 0,
                  MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
              // Src: (intrinsic_wo_chain:{ *:[f32] } 1759:{ *:[iPTR] }, R600_Reg32:{ *:[f32] }:$src0) - Complexity = 8
              // Dst: (RECIPSQRT_IEEE_cm:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 30114*/   0, /*End of Scope*/
/* 30115*/  /*Scope*/ 26, /*->30142*/
/* 30116*/   OPC_CheckChild0Integer, 98|128,50/*6498*/, 
/* 30119*/   OPC_RecordChild1, // #0 = $src0
/* 30120*/   OPC_Scope, 9, /*->30131*/ // 2 children in Scope
/* 30122*/    OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 30124*/    OPC_MorphNodeTo1, TARGET_VAL(R600::CUBE_r600_pseudo), 0,
                  MVT::v4f32, 1/*#Ops*/, 0, 
              // Src: (intrinsic_wo_chain:{ *:[v4f32] } 6498:{ *:[iPTR] }, v4f32:{ *:[v4f32] }:$src0) - Complexity = 8
              // Dst: (CUBE_r600_pseudo:{ *:[v4f32] } v4f32:{ *:[v4f32] }:$src0)
/* 30131*/   /*Scope*/ 9, /*->30141*/
/* 30132*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 30134*/    OPC_MorphNodeTo1, TARGET_VAL(R600::CUBE_eg_pseudo), 0,
                  MVT::v4f32, 1/*#Ops*/, 0, 
              // Src: (intrinsic_wo_chain:{ *:[v4f32] } 6498:{ *:[iPTR] }, v4f32:{ *:[v4f32] }:$src0) - Complexity = 8
              // Dst: (CUBE_eg_pseudo:{ *:[v4f32] } v4f32:{ *:[v4f32] }:$src0)
/* 30141*/   0, /*End of Scope*/
/* 30142*/  0, /*End of Scope*/
/* 30143*/ /*SwitchOpcode*/ 15, TARGET_VAL(AMDGPUISD::CONST_DATA_PTR),// ->30161
/* 30146*/  OPC_RecordChild0, // #0 = $addr
/* 30147*/  OPC_MoveChild0,
/* 30148*/  OPC_CheckOpcode, TARGET_VAL(ISD::TargetGlobalAddress),
/* 30151*/  OPC_MoveParent,
/* 30152*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 30154*/  OPC_MorphNodeTo1, TARGET_VAL(R600::MOV_IMM_GLOBAL_ADDR), 0,
                MVT::iPTR, 1/*#Ops*/, 0, 
            // Src: (AMDGPUconstdata_ptr:{ *:[iPTR] } (tglobaladdr:{ *:[iPTR] }):$addr) - Complexity = 6
            // Dst: (MOV_IMM_GLOBAL_ADDR:{ *:[iPTR] } (tglobaladdr:{ *:[i32] }):$addr)
/* 30161*/ /*SwitchOpcode*/ 20, TARGET_VAL(AMDGPUISD::STORE_MSKOR),// ->30184
/* 30164*/  OPC_RecordMemRef,
/* 30165*/  OPC_RecordNode, // #0 = 'AMDGPUstore_mskor' chained node
/* 30166*/  OPC_RecordChild1, // #1 = $rw_gpr
/* 30167*/  OPC_CheckChild1Type, MVT::v4i32,
/* 30169*/  OPC_RecordChild2, // #2 = $index_gpr
/* 30170*/  OPC_CheckChild2Type, MVT::i32,
/* 30172*/  OPC_CheckPredicate, 24, // Predicate_mskor_global
/* 30174*/  OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 30176*/  OPC_EmitMergeInputChains1_0,
/* 30177*/  OPC_MorphNodeTo0, TARGET_VAL(R600::RAT_MSKOR), 0|OPFL_Chain|OPFL_MemRefs,
                2/*#Ops*/, 1, 2, 
            // Src: (AMDGPUstore_mskor v4i32:{ *:[v4i32] }:$rw_gpr, i32:{ *:[i32] }:$index_gpr)<<P:Predicate_mskor_global>> - Complexity = 4
            // Dst: (RAT_MSKOR v4i32:{ *:[v4i32] }:$rw_gpr, i32:{ *:[i32] }:$index_gpr)
/* 30184*/ /*SwitchOpcode*/ 113, TARGET_VAL(ISD::ATOMIC_SWAP),// ->30300
/* 30187*/  OPC_RecordMemRef,
/* 30188*/  OPC_RecordNode, // #0 = 'atomic_swap' chained node
/* 30189*/  OPC_RecordChild1, // #1 = $ptr
/* 30190*/  OPC_CheckChild1Type, MVT::i32,
/* 30192*/  OPC_RecordChild2, // #2 = $data
/* 30193*/  OPC_CheckType, MVT::i32,
/* 30195*/  OPC_Scope, 42, /*->30239*/ // 2 children in Scope
/* 30197*/   OPC_CheckPredicate, 22, // Predicate_atomic_swap_global_noret_32
/* 30199*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 30201*/   OPC_EmitMergeInputChains1_0,
/* 30202*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::IMPLICIT_DEF), 0,
                 MVT::v4i32, 0/*#Ops*/,  // Results = #3
/* 30208*/   OPC_EmitInteger, MVT::i32, R600::sub0,
/* 30211*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                 MVT::v4i32, 3/*#Ops*/, 3, 2, 4,  // Results = #5
/* 30220*/   OPC_EmitNode1, TARGET_VAL(R600::RAT_ATOMIC_XCHG_INT_NORET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::v4i32, 2/*#Ops*/, 5, 1,  // Results = #6
/* 30228*/   OPC_EmitInteger, MVT::i32, R600::sub1,
/* 30231*/   OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0|OPFL_Chain,
                 MVT::i32, 2/*#Ops*/, 6, 7, 
             // Src: (atomic_swap:{ *:[i32] } i32:{ *:[i32] }:$ptr, i32:{ *:[i32] }:$data)<<P:Predicate_atomic_swap_global_noret_32>> - Complexity = 4
             // Dst: (EXTRACT_SUBREG:{ *:[i32] } (RAT_ATOMIC_XCHG_INT_NORET:{ *:[v4i32] } (INSERT_SUBREG:{ *:[v4i32] } (IMPLICIT_DEF:{ *:[v4i32] }), ?:{ *:[i32] }:$data, sub0:{ *:[i32] }), ?:{ *:[i32] }:$ptr), sub1:{ *:[i32] })
/* 30239*/  /*Scope*/ 59, /*->30299*/
/* 30240*/   OPC_CheckPredicate, 23, // Predicate_atomic_swap_local_32
/* 30242*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 30244*/   OPC_EmitMergeInputChains1_0,
/* 30245*/   OPC_EmitInteger, MVT::i32, 0, 
/* 30248*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 30260*/   OPC_EmitInteger, MVT::i32, 0, 
/* 30263*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 30275*/   OPC_EmitInteger, MVT::i32, 1, 
/* 30278*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 30281*/   OPC_EmitInteger, MVT::i32, 0, 
/* 30284*/   OPC_MorphNodeTo1, TARGET_VAL(R600::LDS_WRXCHG_RET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 9/*#Ops*/, 1, 3, 4, 2, 5, 6, 7, 8, 9, 
             // Src: (atomic_swap:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)<<P:Predicate_atomic_swap_local_32>> - Complexity = 4
             // Dst: (LDS_WRXCHG_RET:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 30299*/  0, /*End of Scope*/
/* 30300*/ /*SwitchOpcode*/ 113, TARGET_VAL(ISD::ATOMIC_LOAD_MIN),// ->30416
/* 30303*/  OPC_RecordMemRef,
/* 30304*/  OPC_RecordNode, // #0 = 'atomic_load_min' chained node
/* 30305*/  OPC_RecordChild1, // #1 = $ptr
/* 30306*/  OPC_CheckChild1Type, MVT::i32,
/* 30308*/  OPC_RecordChild2, // #2 = $data
/* 30309*/  OPC_CheckType, MVT::i32,
/* 30311*/  OPC_Scope, 42, /*->30355*/ // 2 children in Scope
/* 30313*/   OPC_CheckPredicate, 22, // Predicate_atomic_load_min_global_noret_32
/* 30315*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 30317*/   OPC_EmitMergeInputChains1_0,
/* 30318*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::IMPLICIT_DEF), 0,
                 MVT::v4i32, 0/*#Ops*/,  // Results = #3
/* 30324*/   OPC_EmitInteger, MVT::i32, R600::sub0,
/* 30327*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                 MVT::v4i32, 3/*#Ops*/, 3, 2, 4,  // Results = #5
/* 30336*/   OPC_EmitNode1, TARGET_VAL(R600::RAT_ATOMIC_MIN_INT_NORET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::v4i32, 2/*#Ops*/, 5, 1,  // Results = #6
/* 30344*/   OPC_EmitInteger, MVT::i32, R600::sub1,
/* 30347*/   OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0|OPFL_Chain,
                 MVT::i32, 2/*#Ops*/, 6, 7, 
             // Src: (atomic_load_min:{ *:[i32] } i32:{ *:[i32] }:$ptr, i32:{ *:[i32] }:$data)<<P:Predicate_atomic_load_min_global_noret_32>> - Complexity = 4
             // Dst: (EXTRACT_SUBREG:{ *:[i32] } (RAT_ATOMIC_MIN_INT_NORET:{ *:[v4i32] } (INSERT_SUBREG:{ *:[v4i32] } (IMPLICIT_DEF:{ *:[v4i32] }), ?:{ *:[i32] }:$data, sub0:{ *:[i32] }), ?:{ *:[i32] }:$ptr), sub1:{ *:[i32] })
/* 30355*/  /*Scope*/ 59, /*->30415*/
/* 30356*/   OPC_CheckPredicate, 23, // Predicate_atomic_load_min_local_32
/* 30358*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 30360*/   OPC_EmitMergeInputChains1_0,
/* 30361*/   OPC_EmitInteger, MVT::i32, 0, 
/* 30364*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 30376*/   OPC_EmitInteger, MVT::i32, 0, 
/* 30379*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 30391*/   OPC_EmitInteger, MVT::i32, 1, 
/* 30394*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 30397*/   OPC_EmitInteger, MVT::i32, 0, 
/* 30400*/   OPC_MorphNodeTo1, TARGET_VAL(R600::LDS_MIN_INT_RET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 9/*#Ops*/, 1, 3, 4, 2, 5, 6, 7, 8, 9, 
             // Src: (atomic_load_min:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)<<P:Predicate_atomic_load_min_local_32>> - Complexity = 4
             // Dst: (LDS_MIN_INT_RET:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 30415*/  0, /*End of Scope*/
/* 30416*/ /*SwitchOpcode*/ 113, TARGET_VAL(ISD::ATOMIC_LOAD_UMIN),// ->30532
/* 30419*/  OPC_RecordMemRef,
/* 30420*/  OPC_RecordNode, // #0 = 'atomic_load_umin' chained node
/* 30421*/  OPC_RecordChild1, // #1 = $ptr
/* 30422*/  OPC_CheckChild1Type, MVT::i32,
/* 30424*/  OPC_RecordChild2, // #2 = $data
/* 30425*/  OPC_CheckType, MVT::i32,
/* 30427*/  OPC_Scope, 42, /*->30471*/ // 2 children in Scope
/* 30429*/   OPC_CheckPredicate, 22, // Predicate_atomic_load_umin_global_noret_32
/* 30431*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 30433*/   OPC_EmitMergeInputChains1_0,
/* 30434*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::IMPLICIT_DEF), 0,
                 MVT::v4i32, 0/*#Ops*/,  // Results = #3
/* 30440*/   OPC_EmitInteger, MVT::i32, R600::sub0,
/* 30443*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                 MVT::v4i32, 3/*#Ops*/, 3, 2, 4,  // Results = #5
/* 30452*/   OPC_EmitNode1, TARGET_VAL(R600::RAT_ATOMIC_MIN_UINT_NORET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::v4i32, 2/*#Ops*/, 5, 1,  // Results = #6
/* 30460*/   OPC_EmitInteger, MVT::i32, R600::sub1,
/* 30463*/   OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0|OPFL_Chain,
                 MVT::i32, 2/*#Ops*/, 6, 7, 
             // Src: (atomic_load_umin:{ *:[i32] } i32:{ *:[i32] }:$ptr, i32:{ *:[i32] }:$data)<<P:Predicate_atomic_load_umin_global_noret_32>> - Complexity = 4
             // Dst: (EXTRACT_SUBREG:{ *:[i32] } (RAT_ATOMIC_MIN_UINT_NORET:{ *:[v4i32] } (INSERT_SUBREG:{ *:[v4i32] } (IMPLICIT_DEF:{ *:[v4i32] }), ?:{ *:[i32] }:$data, sub0:{ *:[i32] }), ?:{ *:[i32] }:$ptr), sub1:{ *:[i32] })
/* 30471*/  /*Scope*/ 59, /*->30531*/
/* 30472*/   OPC_CheckPredicate, 23, // Predicate_atomic_load_umin_local_32
/* 30474*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 30476*/   OPC_EmitMergeInputChains1_0,
/* 30477*/   OPC_EmitInteger, MVT::i32, 0, 
/* 30480*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 30492*/   OPC_EmitInteger, MVT::i32, 0, 
/* 30495*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 30507*/   OPC_EmitInteger, MVT::i32, 1, 
/* 30510*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 30513*/   OPC_EmitInteger, MVT::i32, 0, 
/* 30516*/   OPC_MorphNodeTo1, TARGET_VAL(R600::LDS_MIN_UINT_RET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 9/*#Ops*/, 1, 3, 4, 2, 5, 6, 7, 8, 9, 
             // Src: (atomic_load_umin:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)<<P:Predicate_atomic_load_umin_local_32>> - Complexity = 4
             // Dst: (LDS_MIN_UINT_RET:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 30531*/  0, /*End of Scope*/
/* 30532*/ /*SwitchOpcode*/ 113, TARGET_VAL(ISD::ATOMIC_LOAD_MAX),// ->30648
/* 30535*/  OPC_RecordMemRef,
/* 30536*/  OPC_RecordNode, // #0 = 'atomic_load_max' chained node
/* 30537*/  OPC_RecordChild1, // #1 = $ptr
/* 30538*/  OPC_CheckChild1Type, MVT::i32,
/* 30540*/  OPC_RecordChild2, // #2 = $data
/* 30541*/  OPC_CheckType, MVT::i32,
/* 30543*/  OPC_Scope, 42, /*->30587*/ // 2 children in Scope
/* 30545*/   OPC_CheckPredicate, 22, // Predicate_atomic_load_max_global_noret_32
/* 30547*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 30549*/   OPC_EmitMergeInputChains1_0,
/* 30550*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::IMPLICIT_DEF), 0,
                 MVT::v4i32, 0/*#Ops*/,  // Results = #3
/* 30556*/   OPC_EmitInteger, MVT::i32, R600::sub0,
/* 30559*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                 MVT::v4i32, 3/*#Ops*/, 3, 2, 4,  // Results = #5
/* 30568*/   OPC_EmitNode1, TARGET_VAL(R600::RAT_ATOMIC_MAX_INT_NORET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::v4i32, 2/*#Ops*/, 5, 1,  // Results = #6
/* 30576*/   OPC_EmitInteger, MVT::i32, R600::sub1,
/* 30579*/   OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0|OPFL_Chain,
                 MVT::i32, 2/*#Ops*/, 6, 7, 
             // Src: (atomic_load_max:{ *:[i32] } i32:{ *:[i32] }:$ptr, i32:{ *:[i32] }:$data)<<P:Predicate_atomic_load_max_global_noret_32>> - Complexity = 4
             // Dst: (EXTRACT_SUBREG:{ *:[i32] } (RAT_ATOMIC_MAX_INT_NORET:{ *:[v4i32] } (INSERT_SUBREG:{ *:[v4i32] } (IMPLICIT_DEF:{ *:[v4i32] }), ?:{ *:[i32] }:$data, sub0:{ *:[i32] }), ?:{ *:[i32] }:$ptr), sub1:{ *:[i32] })
/* 30587*/  /*Scope*/ 59, /*->30647*/
/* 30588*/   OPC_CheckPredicate, 23, // Predicate_atomic_load_max_local_32
/* 30590*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 30592*/   OPC_EmitMergeInputChains1_0,
/* 30593*/   OPC_EmitInteger, MVT::i32, 0, 
/* 30596*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 30608*/   OPC_EmitInteger, MVT::i32, 0, 
/* 30611*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 30623*/   OPC_EmitInteger, MVT::i32, 1, 
/* 30626*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 30629*/   OPC_EmitInteger, MVT::i32, 0, 
/* 30632*/   OPC_MorphNodeTo1, TARGET_VAL(R600::LDS_MAX_INT_RET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 9/*#Ops*/, 1, 3, 4, 2, 5, 6, 7, 8, 9, 
             // Src: (atomic_load_max:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)<<P:Predicate_atomic_load_max_local_32>> - Complexity = 4
             // Dst: (LDS_MAX_INT_RET:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 30647*/  0, /*End of Scope*/
/* 30648*/ /*SwitchOpcode*/ 113, TARGET_VAL(ISD::ATOMIC_LOAD_UMAX),// ->30764
/* 30651*/  OPC_RecordMemRef,
/* 30652*/  OPC_RecordNode, // #0 = 'atomic_load_umax' chained node
/* 30653*/  OPC_RecordChild1, // #1 = $ptr
/* 30654*/  OPC_CheckChild1Type, MVT::i32,
/* 30656*/  OPC_RecordChild2, // #2 = $data
/* 30657*/  OPC_CheckType, MVT::i32,
/* 30659*/  OPC_Scope, 42, /*->30703*/ // 2 children in Scope
/* 30661*/   OPC_CheckPredicate, 22, // Predicate_atomic_load_umax_global_noret_32
/* 30663*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 30665*/   OPC_EmitMergeInputChains1_0,
/* 30666*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::IMPLICIT_DEF), 0,
                 MVT::v4i32, 0/*#Ops*/,  // Results = #3
/* 30672*/   OPC_EmitInteger, MVT::i32, R600::sub0,
/* 30675*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                 MVT::v4i32, 3/*#Ops*/, 3, 2, 4,  // Results = #5
/* 30684*/   OPC_EmitNode1, TARGET_VAL(R600::RAT_ATOMIC_MAX_UINT_NORET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::v4i32, 2/*#Ops*/, 5, 1,  // Results = #6
/* 30692*/   OPC_EmitInteger, MVT::i32, R600::sub1,
/* 30695*/   OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0|OPFL_Chain,
                 MVT::i32, 2/*#Ops*/, 6, 7, 
             // Src: (atomic_load_umax:{ *:[i32] } i32:{ *:[i32] }:$ptr, i32:{ *:[i32] }:$data)<<P:Predicate_atomic_load_umax_global_noret_32>> - Complexity = 4
             // Dst: (EXTRACT_SUBREG:{ *:[i32] } (RAT_ATOMIC_MAX_UINT_NORET:{ *:[v4i32] } (INSERT_SUBREG:{ *:[v4i32] } (IMPLICIT_DEF:{ *:[v4i32] }), ?:{ *:[i32] }:$data, sub0:{ *:[i32] }), ?:{ *:[i32] }:$ptr), sub1:{ *:[i32] })
/* 30703*/  /*Scope*/ 59, /*->30763*/
/* 30704*/   OPC_CheckPredicate, 23, // Predicate_atomic_load_umax_local_32
/* 30706*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 30708*/   OPC_EmitMergeInputChains1_0,
/* 30709*/   OPC_EmitInteger, MVT::i32, 0, 
/* 30712*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 30724*/   OPC_EmitInteger, MVT::i32, 0, 
/* 30727*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 30739*/   OPC_EmitInteger, MVT::i32, 1, 
/* 30742*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 30745*/   OPC_EmitInteger, MVT::i32, 0, 
/* 30748*/   OPC_MorphNodeTo1, TARGET_VAL(R600::LDS_MAX_UINT_RET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 9/*#Ops*/, 1, 3, 4, 2, 5, 6, 7, 8, 9, 
             // Src: (atomic_load_umax:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)<<P:Predicate_atomic_load_umax_local_32>> - Complexity = 4
             // Dst: (LDS_MAX_UINT_RET:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 30763*/  0, /*End of Scope*/
/* 30764*/ /*SwitchOpcode*/ 113, TARGET_VAL(ISD::ATOMIC_LOAD_AND),// ->30880
/* 30767*/  OPC_RecordMemRef,
/* 30768*/  OPC_RecordNode, // #0 = 'atomic_load_and' chained node
/* 30769*/  OPC_RecordChild1, // #1 = $ptr
/* 30770*/  OPC_CheckChild1Type, MVT::i32,
/* 30772*/  OPC_RecordChild2, // #2 = $data
/* 30773*/  OPC_CheckType, MVT::i32,
/* 30775*/  OPC_Scope, 42, /*->30819*/ // 2 children in Scope
/* 30777*/   OPC_CheckPredicate, 22, // Predicate_atomic_load_and_global_noret_32
/* 30779*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 30781*/   OPC_EmitMergeInputChains1_0,
/* 30782*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::IMPLICIT_DEF), 0,
                 MVT::v4i32, 0/*#Ops*/,  // Results = #3
/* 30788*/   OPC_EmitInteger, MVT::i32, R600::sub0,
/* 30791*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                 MVT::v4i32, 3/*#Ops*/, 3, 2, 4,  // Results = #5
/* 30800*/   OPC_EmitNode1, TARGET_VAL(R600::RAT_ATOMIC_AND_NORET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::v4i32, 2/*#Ops*/, 5, 1,  // Results = #6
/* 30808*/   OPC_EmitInteger, MVT::i32, R600::sub1,
/* 30811*/   OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0|OPFL_Chain,
                 MVT::i32, 2/*#Ops*/, 6, 7, 
             // Src: (atomic_load_and:{ *:[i32] } i32:{ *:[i32] }:$ptr, i32:{ *:[i32] }:$data)<<P:Predicate_atomic_load_and_global_noret_32>> - Complexity = 4
             // Dst: (EXTRACT_SUBREG:{ *:[i32] } (RAT_ATOMIC_AND_NORET:{ *:[v4i32] } (INSERT_SUBREG:{ *:[v4i32] } (IMPLICIT_DEF:{ *:[v4i32] }), ?:{ *:[i32] }:$data, sub0:{ *:[i32] }), ?:{ *:[i32] }:$ptr), sub1:{ *:[i32] })
/* 30819*/  /*Scope*/ 59, /*->30879*/
/* 30820*/   OPC_CheckPredicate, 23, // Predicate_atomic_load_and_local_32
/* 30822*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 30824*/   OPC_EmitMergeInputChains1_0,
/* 30825*/   OPC_EmitInteger, MVT::i32, 0, 
/* 30828*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 30840*/   OPC_EmitInteger, MVT::i32, 0, 
/* 30843*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 30855*/   OPC_EmitInteger, MVT::i32, 1, 
/* 30858*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 30861*/   OPC_EmitInteger, MVT::i32, 0, 
/* 30864*/   OPC_MorphNodeTo1, TARGET_VAL(R600::LDS_AND_RET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 9/*#Ops*/, 1, 3, 4, 2, 5, 6, 7, 8, 9, 
             // Src: (atomic_load_and:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)<<P:Predicate_atomic_load_and_local_32>> - Complexity = 4
             // Dst: (LDS_AND_RET:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 30879*/  0, /*End of Scope*/
/* 30880*/ /*SwitchOpcode*/ 113, TARGET_VAL(ISD::ATOMIC_LOAD_OR),// ->30996
/* 30883*/  OPC_RecordMemRef,
/* 30884*/  OPC_RecordNode, // #0 = 'atomic_load_or' chained node
/* 30885*/  OPC_RecordChild1, // #1 = $ptr
/* 30886*/  OPC_CheckChild1Type, MVT::i32,
/* 30888*/  OPC_RecordChild2, // #2 = $data
/* 30889*/  OPC_CheckType, MVT::i32,
/* 30891*/  OPC_Scope, 42, /*->30935*/ // 2 children in Scope
/* 30893*/   OPC_CheckPredicate, 22, // Predicate_atomic_load_or_global_noret_32
/* 30895*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 30897*/   OPC_EmitMergeInputChains1_0,
/* 30898*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::IMPLICIT_DEF), 0,
                 MVT::v4i32, 0/*#Ops*/,  // Results = #3
/* 30904*/   OPC_EmitInteger, MVT::i32, R600::sub0,
/* 30907*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                 MVT::v4i32, 3/*#Ops*/, 3, 2, 4,  // Results = #5
/* 30916*/   OPC_EmitNode1, TARGET_VAL(R600::RAT_ATOMIC_OR_NORET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::v4i32, 2/*#Ops*/, 5, 1,  // Results = #6
/* 30924*/   OPC_EmitInteger, MVT::i32, R600::sub1,
/* 30927*/   OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0|OPFL_Chain,
                 MVT::i32, 2/*#Ops*/, 6, 7, 
             // Src: (atomic_load_or:{ *:[i32] } i32:{ *:[i32] }:$ptr, i32:{ *:[i32] }:$data)<<P:Predicate_atomic_load_or_global_noret_32>> - Complexity = 4
             // Dst: (EXTRACT_SUBREG:{ *:[i32] } (RAT_ATOMIC_OR_NORET:{ *:[v4i32] } (INSERT_SUBREG:{ *:[v4i32] } (IMPLICIT_DEF:{ *:[v4i32] }), ?:{ *:[i32] }:$data, sub0:{ *:[i32] }), ?:{ *:[i32] }:$ptr), sub1:{ *:[i32] })
/* 30935*/  /*Scope*/ 59, /*->30995*/
/* 30936*/   OPC_CheckPredicate, 23, // Predicate_atomic_load_or_local_32
/* 30938*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 30940*/   OPC_EmitMergeInputChains1_0,
/* 30941*/   OPC_EmitInteger, MVT::i32, 0, 
/* 30944*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 30956*/   OPC_EmitInteger, MVT::i32, 0, 
/* 30959*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 30971*/   OPC_EmitInteger, MVT::i32, 1, 
/* 30974*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 30977*/   OPC_EmitInteger, MVT::i32, 0, 
/* 30980*/   OPC_MorphNodeTo1, TARGET_VAL(R600::LDS_OR_RET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 9/*#Ops*/, 1, 3, 4, 2, 5, 6, 7, 8, 9, 
             // Src: (atomic_load_or:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)<<P:Predicate_atomic_load_or_local_32>> - Complexity = 4
             // Dst: (LDS_OR_RET:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 30995*/  0, /*End of Scope*/
/* 30996*/ /*SwitchOpcode*/ 113, TARGET_VAL(ISD::ATOMIC_LOAD_XOR),// ->31112
/* 30999*/  OPC_RecordMemRef,
/* 31000*/  OPC_RecordNode, // #0 = 'atomic_load_xor' chained node
/* 31001*/  OPC_RecordChild1, // #1 = $ptr
/* 31002*/  OPC_CheckChild1Type, MVT::i32,
/* 31004*/  OPC_RecordChild2, // #2 = $data
/* 31005*/  OPC_CheckType, MVT::i32,
/* 31007*/  OPC_Scope, 42, /*->31051*/ // 2 children in Scope
/* 31009*/   OPC_CheckPredicate, 22, // Predicate_atomic_load_xor_global_noret_32
/* 31011*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 31013*/   OPC_EmitMergeInputChains1_0,
/* 31014*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::IMPLICIT_DEF), 0,
                 MVT::v4i32, 0/*#Ops*/,  // Results = #3
/* 31020*/   OPC_EmitInteger, MVT::i32, R600::sub0,
/* 31023*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                 MVT::v4i32, 3/*#Ops*/, 3, 2, 4,  // Results = #5
/* 31032*/   OPC_EmitNode1, TARGET_VAL(R600::RAT_ATOMIC_XOR_NORET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::v4i32, 2/*#Ops*/, 5, 1,  // Results = #6
/* 31040*/   OPC_EmitInteger, MVT::i32, R600::sub1,
/* 31043*/   OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0|OPFL_Chain,
                 MVT::i32, 2/*#Ops*/, 6, 7, 
             // Src: (atomic_load_xor:{ *:[i32] } i32:{ *:[i32] }:$ptr, i32:{ *:[i32] }:$data)<<P:Predicate_atomic_load_xor_global_noret_32>> - Complexity = 4
             // Dst: (EXTRACT_SUBREG:{ *:[i32] } (RAT_ATOMIC_XOR_NORET:{ *:[v4i32] } (INSERT_SUBREG:{ *:[v4i32] } (IMPLICIT_DEF:{ *:[v4i32] }), ?:{ *:[i32] }:$data, sub0:{ *:[i32] }), ?:{ *:[i32] }:$ptr), sub1:{ *:[i32] })
/* 31051*/  /*Scope*/ 59, /*->31111*/
/* 31052*/   OPC_CheckPredicate, 23, // Predicate_atomic_load_xor_local_32
/* 31054*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 31056*/   OPC_EmitMergeInputChains1_0,
/* 31057*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31060*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 31072*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31075*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 31087*/   OPC_EmitInteger, MVT::i32, 1, 
/* 31090*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 31093*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31096*/   OPC_MorphNodeTo1, TARGET_VAL(R600::LDS_XOR_RET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 9/*#Ops*/, 1, 3, 4, 2, 5, 6, 7, 8, 9, 
             // Src: (atomic_load_xor:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)<<P:Predicate_atomic_load_xor_local_32>> - Complexity = 4
             // Dst: (LDS_XOR_RET:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 31111*/  0, /*End of Scope*/
/* 31112*/ /*SwitchOpcode*/ 16|128,1/*144*/, TARGET_VAL(ISD::ATOMIC_CMP_SWAP),// ->31260
/* 31116*/  OPC_RecordMemRef,
/* 31117*/  OPC_RecordNode, // #0 = 'atomic_cmp_swap' chained node
/* 31118*/  OPC_RecordChild1, // #1 = $ptr
/* 31119*/  OPC_CheckChild1Type, MVT::i32,
/* 31121*/  OPC_RecordChild2, // #2 = $cmp
/* 31122*/  OPC_RecordChild3, // #3 = $data
/* 31123*/  OPC_CheckType, MVT::i32,
/* 31125*/  OPC_Scope, 54, /*->31181*/ // 2 children in Scope
/* 31127*/   OPC_CheckPredicate, 25, // Predicate_atomic_cmp_swap_global_noret
/* 31129*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 31131*/   OPC_EmitMergeInputChains1_0,
/* 31132*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::IMPLICIT_DEF), 0,
                 MVT::v4i32, 0/*#Ops*/,  // Results = #4
/* 31138*/   OPC_EmitInteger, MVT::i32, R600::sub3,
/* 31141*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                 MVT::v4i32, 3/*#Ops*/, 4, 2, 5,  // Results = #6
/* 31150*/   OPC_EmitInteger, MVT::i32, R600::sub0,
/* 31153*/   OPC_EmitNode1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                 MVT::v4i32, 3/*#Ops*/, 6, 3, 7,  // Results = #8
/* 31162*/   OPC_EmitNode1, TARGET_VAL(R600::RAT_ATOMIC_CMPXCHG_INT_NORET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::v4i32, 2/*#Ops*/, 8, 1,  // Results = #9
/* 31170*/   OPC_EmitInteger, MVT::i32, R600::sub1,
/* 31173*/   OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0|OPFL_Chain,
                 MVT::i32, 2/*#Ops*/, 9, 10, 
             // Src: (atomic_cmp_swap:{ *:[i32] } i32:{ *:[i32] }:$ptr, i32:{ *:[i32] }:$cmp, i32:{ *:[i32] }:$data)<<P:Predicate_atomic_cmp_swap_global_noret>> - Complexity = 4
             // Dst: (EXTRACT_SUBREG:{ *:[i32] } (RAT_ATOMIC_CMPXCHG_INT_NORET:{ *:[v4i32] } (INSERT_SUBREG:{ *:[v4i32] } (INSERT_SUBREG:{ *:[v4i32] } (IMPLICIT_DEF:{ *:[v4i32] }), ?:{ *:[i32] }:$cmp, sub3:{ *:[i32] }), ?:{ *:[i32] }:$data, sub0:{ *:[i32] }), ?:{ *:[i32] }:$ptr), sub1:{ *:[i32] })
/* 31181*/  /*Scope*/ 77, /*->31259*/
/* 31182*/   OPC_CheckPredicate, 23, // Predicate_atomic_cmp_swap_local_32
/* 31184*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 31186*/   OPC_EmitMergeInputChains1_0,
/* 31187*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31190*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 31202*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31205*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 31217*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31220*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 31232*/   OPC_EmitInteger, MVT::i32, 1, 
/* 31235*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 31238*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31241*/   OPC_MorphNodeTo1, TARGET_VAL(R600::LDS_CMPST_RET), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 12/*#Ops*/, 1, 4, 5, 2, 6, 7, 3, 8, 9, 10, 11, 12, 
             // Src: (atomic_cmp_swap:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2)<<P:Predicate_atomic_cmp_swap_local_32>> - Complexity = 4
             // Dst: (LDS_CMPST_RET:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2)
/* 31259*/  0, /*End of Scope*/
/* 31260*/ /*SwitchOpcode*/ 57, TARGET_VAL(ISD::BITCAST),// ->31320
/* 31263*/  OPC_RecordChild0, // #0 = $src0
/* 31264*/  OPC_SwitchType /*6 cases */, 7, MVT::i32,// ->31274
/* 31267*/   OPC_CheckChild0Type, MVT::f32,
/* 31269*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 31271*/   OPC_CompleteMatch, 1, 0, 
              // Src: (bitconvert:{ *:[i32] } R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
              // Dst: R600_Reg32:{ *:[i32] }:$src0
/* 31274*/  /*SwitchType*/ 7, MVT::f32,// ->31283
/* 31276*/   OPC_CheckChild0Type, MVT::i32,
/* 31278*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 31280*/   OPC_CompleteMatch, 1, 0, 
              // Src: (bitconvert:{ *:[f32] } R600_Reg32:{ *:[i32] }:$src0) - Complexity = 3
              // Dst: R600_Reg32:{ *:[f32] }:$src0
/* 31283*/  /*SwitchType*/ 7, MVT::v2i32,// ->31292
/* 31285*/   OPC_CheckChild0Type, MVT::v2f32,
/* 31287*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 31289*/   OPC_CompleteMatch, 1, 0, 
              // Src: (bitconvert:{ *:[v2i32] } R600_Reg64:{ *:[v2f32] }:$src0) - Complexity = 3
              // Dst: R600_Reg64:{ *:[v2i32] }:$src0
/* 31292*/  /*SwitchType*/ 7, MVT::v4i32,// ->31301
/* 31294*/   OPC_CheckChild0Type, MVT::v4f32,
/* 31296*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 31298*/   OPC_CompleteMatch, 1, 0, 
              // Src: (bitconvert:{ *:[v4i32] } R600_Reg128:{ *:[v4f32] }:$src0) - Complexity = 3
              // Dst: R600_Reg128:{ *:[v4i32] }:$src0
/* 31301*/  /*SwitchType*/ 7, MVT::v2f32,// ->31310
/* 31303*/   OPC_CheckChild0Type, MVT::v2i32,
/* 31305*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 31307*/   OPC_CompleteMatch, 1, 0, 
              // Src: (bitconvert:{ *:[v2f32] } R600_Reg64:{ *:[v2i32] }:$src0) - Complexity = 3
              // Dst: R600_Reg64:{ *:[v2f32] }:$src0
/* 31310*/  /*SwitchType*/ 7, MVT::v4f32,// ->31319
/* 31312*/   OPC_CheckChild0Type, MVT::v4i32,
/* 31314*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 31316*/   OPC_CompleteMatch, 1, 0, 
              // Src: (bitconvert:{ *:[v4f32] } R600_Reg128:{ *:[v4i32] }:$src0) - Complexity = 3
              // Dst: R600_Reg128:{ *:[v4f32] }:$src0
/* 31319*/  0, // EndSwitchType
/* 31320*/ /*SwitchOpcode*/ 8, TARGET_VAL(AMDGPUISD::DWORDADDR),// ->31331
/* 31323*/  OPC_RecordChild0, // #0 = $addr
/* 31324*/  OPC_CheckType, MVT::i32,
/* 31326*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 31328*/  OPC_CompleteMatch, 1, 0, 
             // Src: (AMDGPUdwordaddr:{ *:[i32] } R600_Reg32:{ *:[i32] }:$addr) - Complexity = 3
             // Dst: R600_Reg32:{ *:[i32] }:$addr
/* 31331*/ /*SwitchOpcode*/ 9, TARGET_VAL(AMDGPUISD::DUMMY_CHAIN),// ->31343
/* 31334*/  OPC_RecordNode, // #0 = 'R600dummy_chain' chained node
/* 31335*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 31337*/  OPC_EmitMergeInputChains1_0,
/* 31338*/  OPC_MorphNodeTo0, TARGET_VAL(R600::DUMMY_CHAIN), 0|OPFL_Chain,
                0/*#Ops*/, 
            // Src: (R600dummy_chain) - Complexity = 3
            // Dst: (DUMMY_CHAIN)
/* 31343*/ /*SwitchOpcode*/ 104, TARGET_VAL(ISD::SUB),// ->31450
/* 31346*/  OPC_RecordChild0, // #0 = $src0
/* 31347*/  OPC_RecordChild1, // #1 = $src1
/* 31348*/  OPC_CheckType, MVT::i32,
/* 31350*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 31352*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31355*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31358*/  OPC_EmitInteger, MVT::i32, 1, 
/* 31361*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31364*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31367*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31370*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31373*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31376*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31379*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 31391*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31394*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31397*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31400*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 31412*/  OPC_EmitInteger, MVT::i32, 1, 
/* 31415*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 31418*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31421*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31424*/  OPC_MorphNodeTo1, TARGET_VAL(R600::SUB_INT), 0,
                MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
            // Src: (sub:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
            // Dst: (SUB_INT:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 31450*/ /*SwitchOpcode*/ 104, TARGET_VAL(ISD::SMAX),// ->31557
/* 31453*/  OPC_RecordChild0, // #0 = $src0
/* 31454*/  OPC_RecordChild1, // #1 = $src1
/* 31455*/  OPC_CheckType, MVT::i32,
/* 31457*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 31459*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31462*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31465*/  OPC_EmitInteger, MVT::i32, 1, 
/* 31468*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31471*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31474*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31477*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31480*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31483*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31486*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 31498*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31501*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31504*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31507*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 31519*/  OPC_EmitInteger, MVT::i32, 1, 
/* 31522*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 31525*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31528*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31531*/  OPC_MorphNodeTo1, TARGET_VAL(R600::MAX_INT), 0,
                MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
            // Src: (smax:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
            // Dst: (MAX_INT:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 31557*/ /*SwitchOpcode*/ 104, TARGET_VAL(ISD::SMIN),// ->31664
/* 31560*/  OPC_RecordChild0, // #0 = $src0
/* 31561*/  OPC_RecordChild1, // #1 = $src1
/* 31562*/  OPC_CheckType, MVT::i32,
/* 31564*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 31566*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31569*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31572*/  OPC_EmitInteger, MVT::i32, 1, 
/* 31575*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31578*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31581*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31584*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31587*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31590*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31593*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 31605*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31608*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31611*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31614*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 31626*/  OPC_EmitInteger, MVT::i32, 1, 
/* 31629*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 31632*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31635*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31638*/  OPC_MorphNodeTo1, TARGET_VAL(R600::MIN_INT), 0,
                MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
            // Src: (smin:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
            // Dst: (MIN_INT:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 31664*/ /*SwitchOpcode*/ 104, TARGET_VAL(ISD::UMAX),// ->31771
/* 31667*/  OPC_RecordChild0, // #0 = $src0
/* 31668*/  OPC_RecordChild1, // #1 = $src1
/* 31669*/  OPC_CheckType, MVT::i32,
/* 31671*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 31673*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31676*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31679*/  OPC_EmitInteger, MVT::i32, 1, 
/* 31682*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31685*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31688*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31691*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31694*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31697*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31700*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 31712*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31715*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31718*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31721*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 31733*/  OPC_EmitInteger, MVT::i32, 1, 
/* 31736*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 31739*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31742*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31745*/  OPC_MorphNodeTo1, TARGET_VAL(R600::MAX_UINT), 0,
                MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
            // Src: (umax:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
            // Dst: (MAX_UINT:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 31771*/ /*SwitchOpcode*/ 104, TARGET_VAL(ISD::UMIN),// ->31878
/* 31774*/  OPC_RecordChild0, // #0 = $src0
/* 31775*/  OPC_RecordChild1, // #1 = $src1
/* 31776*/  OPC_CheckType, MVT::i32,
/* 31778*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 31780*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31783*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31786*/  OPC_EmitInteger, MVT::i32, 1, 
/* 31789*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31792*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31795*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31798*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31801*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31804*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31807*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 31819*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31822*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31825*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31828*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 31840*/  OPC_EmitInteger, MVT::i32, 1, 
/* 31843*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 31846*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31849*/  OPC_EmitInteger, MVT::i32, 0, 
/* 31852*/  OPC_MorphNodeTo1, TARGET_VAL(R600::MIN_UINT), 0,
                MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
            // Src: (umin:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
            // Dst: (MIN_UINT:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 31878*/ /*SwitchOpcode*/ 78|128,1/*206*/, TARGET_VAL(ISD::FP_TO_SINT),// ->32088
/* 31882*/  OPC_RecordChild0, // #0 = $src0
/* 31883*/  OPC_CheckChild0Type, MVT::f32,
/* 31885*/  OPC_CheckType, MVT::i32,
/* 31887*/  OPC_Scope, 66, /*->31955*/ // 2 children in Scope
/* 31889*/   OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 31891*/   OPC_EmitInteger, MVT::i32, 1, 
/* 31894*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31897*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31900*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31903*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31906*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31909*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31912*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 31924*/   OPC_EmitInteger, MVT::i32, 1, 
/* 31927*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 31930*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31933*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31936*/   OPC_MorphNodeTo1, TARGET_VAL(R600::FLT_TO_INT_r600), 0,
                 MVT::i32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (fp_to_sint:{ *:[i32] } R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (FLT_TO_INT_r600:{ *:[i32] } R600_Reg32:{ *:[f32] }:$src0)
/* 31955*/  /*Scope*/ 2|128,1/*130*/, /*->32087*/
/* 31957*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 31959*/   OPC_EmitInteger, MVT::i32, 1, 
/* 31962*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31965*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31968*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31971*/   OPC_EmitInteger, MVT::i32, 1, 
/* 31974*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31977*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31980*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31983*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31986*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31989*/   OPC_EmitInteger, MVT::i32, 0, 
/* 31992*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 32004*/   OPC_EmitInteger, MVT::i32, 1, 
/* 32007*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 32010*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32013*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32016*/   OPC_EmitNode1, TARGET_VAL(R600::TRUNC), 0,
                 MVT::i32, 13/*#Ops*/, 5, 6, 7, 8, 0, 9, 10, 11, 12, 13, 14, 15, 16,  // Results = #17
/* 32035*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32038*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32041*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32044*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 32056*/   OPC_EmitInteger, MVT::i32, 1, 
/* 32059*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 32062*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32065*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32068*/   OPC_MorphNodeTo1, TARGET_VAL(R600::FLT_TO_INT_eg), 0,
                 MVT::i32, 13/*#Ops*/, 1, 2, 3, 4, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
             // Src: (fp_to_sint:{ *:[i32] } f32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (FLT_TO_INT_eg:{ *:[i32] } (TRUNC:{ *:[i32] } ?:{ *:[f32] }:$src0))
/* 32087*/  0, /*End of Scope*/
/* 32088*/ /*SwitchOpcode*/ 78|128,1/*206*/, TARGET_VAL(ISD::FP_TO_UINT),// ->32298
/* 32092*/  OPC_RecordChild0, // #0 = $src0
/* 32093*/  OPC_CheckChild0Type, MVT::f32,
/* 32095*/  OPC_CheckType, MVT::i32,
/* 32097*/  OPC_Scope, 66, /*->32165*/ // 2 children in Scope
/* 32099*/   OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 32101*/   OPC_EmitInteger, MVT::i32, 1, 
/* 32104*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32107*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32110*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32113*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32116*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32119*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32122*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 32134*/   OPC_EmitInteger, MVT::i32, 1, 
/* 32137*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 32140*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32143*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32146*/   OPC_MorphNodeTo1, TARGET_VAL(R600::FLT_TO_UINT_r600), 0,
                 MVT::i32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (fp_to_uint:{ *:[i32] } R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (FLT_TO_UINT_r600:{ *:[i32] } R600_Reg32:{ *:[f32] }:$src0)
/* 32165*/  /*Scope*/ 2|128,1/*130*/, /*->32297*/
/* 32167*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 32169*/   OPC_EmitInteger, MVT::i32, 1, 
/* 32172*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32175*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32178*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32181*/   OPC_EmitInteger, MVT::i32, 1, 
/* 32184*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32187*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32190*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32193*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32196*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32199*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32202*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 32214*/   OPC_EmitInteger, MVT::i32, 1, 
/* 32217*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 32220*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32223*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32226*/   OPC_EmitNode1, TARGET_VAL(R600::TRUNC), 0,
                 MVT::i32, 13/*#Ops*/, 5, 6, 7, 8, 0, 9, 10, 11, 12, 13, 14, 15, 16,  // Results = #17
/* 32245*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32248*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32251*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32254*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 32266*/   OPC_EmitInteger, MVT::i32, 1, 
/* 32269*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 32272*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32275*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32278*/   OPC_MorphNodeTo1, TARGET_VAL(R600::FLT_TO_UINT_eg), 0,
                 MVT::i32, 13/*#Ops*/, 1, 2, 3, 4, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
             // Src: (fp_to_uint:{ *:[i32] } f32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (FLT_TO_UINT_eg:{ *:[i32] } (TRUNC:{ *:[i32] } ?:{ *:[f32] }:$src0))
/* 32297*/  0, /*End of Scope*/
/* 32298*/ /*SwitchOpcode*/ 82|128,1/*210*/, TARGET_VAL(ISD::SHL),// ->32512
/* 32302*/  OPC_RecordChild0, // #0 = $src0
/* 32303*/  OPC_RecordChild1, // #1 = $src1
/* 32304*/  OPC_CheckChild1Type, MVT::i32,
/* 32306*/  OPC_CheckType, MVT::i32,
/* 32308*/  OPC_Scope, 100, /*->32410*/ // 2 children in Scope
/* 32310*/   OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 32312*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32315*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32318*/   OPC_EmitInteger, MVT::i32, 1, 
/* 32321*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32324*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32327*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32330*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32333*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32336*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32339*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 32351*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32354*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32357*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32360*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 32372*/   OPC_EmitInteger, MVT::i32, 1, 
/* 32375*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 32378*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32381*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32384*/   OPC_MorphNodeTo1, TARGET_VAL(R600::LSHL_r600), 0,
                 MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
             // Src: (shl:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
             // Dst: (LSHL_r600:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 32410*/  /*Scope*/ 100, /*->32511*/
/* 32411*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 32413*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32416*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32419*/   OPC_EmitInteger, MVT::i32, 1, 
/* 32422*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32425*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32428*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32431*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32434*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32437*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32440*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 32452*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32455*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32458*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32461*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 32473*/   OPC_EmitInteger, MVT::i32, 1, 
/* 32476*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 32479*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32482*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32485*/   OPC_MorphNodeTo1, TARGET_VAL(R600::LSHL_eg), 0,
                 MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
             // Src: (shl:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
             // Dst: (LSHL_eg:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 32511*/  0, /*End of Scope*/
/* 32512*/ /*SwitchOpcode*/ 53|128,2/*309*/, TARGET_VAL(ISD::MUL),// ->32825
/* 32516*/  OPC_RecordChild0, // #0 = $src0
/* 32517*/  OPC_RecordChild1, // #1 = $src1
/* 32518*/  OPC_CheckType, MVT::i32,
/* 32520*/  OPC_Scope, 100, /*->32622*/ // 3 children in Scope
/* 32522*/   OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 32524*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32527*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32530*/   OPC_EmitInteger, MVT::i32, 1, 
/* 32533*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32536*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32539*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32542*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32545*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32548*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32551*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 32563*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32566*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32569*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32572*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 32584*/   OPC_EmitInteger, MVT::i32, 1, 
/* 32587*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 32590*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32593*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32596*/   OPC_MorphNodeTo1, TARGET_VAL(R600::MULLO_INT_r600), 0,
                 MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
             // Src: (mul:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
             // Dst: (MULLO_INT_r600:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 32622*/  /*Scope*/ 100, /*->32723*/
/* 32623*/   OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 32625*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32628*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32631*/   OPC_EmitInteger, MVT::i32, 1, 
/* 32634*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32637*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32640*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32643*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32646*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32649*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32652*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 32664*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32667*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32670*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32673*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 32685*/   OPC_EmitInteger, MVT::i32, 1, 
/* 32688*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 32691*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32694*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32697*/   OPC_MorphNodeTo1, TARGET_VAL(R600::MULLO_INT_eg), 0,
                 MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
             // Src: (mul:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
             // Dst: (MULLO_INT_eg:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 32723*/  /*Scope*/ 100, /*->32824*/
/* 32724*/   OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 32726*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32729*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32732*/   OPC_EmitInteger, MVT::i32, 1, 
/* 32735*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32738*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32741*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32744*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32747*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32750*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32753*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 32765*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32768*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32771*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32774*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 32786*/   OPC_EmitInteger, MVT::i32, 1, 
/* 32789*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 32792*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32795*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32798*/   OPC_MorphNodeTo1, TARGET_VAL(R600::MULLO_INT_cm), 0,
                 MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
             // Src: (mul:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
             // Dst: (MULLO_INT_cm:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 32824*/  0, /*End of Scope*/
/* 32825*/ /*SwitchOpcode*/ 53|128,2/*309*/, TARGET_VAL(ISD::MULHS),// ->33138
/* 32829*/  OPC_RecordChild0, // #0 = $src0
/* 32830*/  OPC_RecordChild1, // #1 = $src1
/* 32831*/  OPC_CheckType, MVT::i32,
/* 32833*/  OPC_Scope, 100, /*->32935*/ // 3 children in Scope
/* 32835*/   OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 32837*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32840*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32843*/   OPC_EmitInteger, MVT::i32, 1, 
/* 32846*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32849*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32852*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32855*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32858*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32861*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32864*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 32876*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32879*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32882*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32885*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 32897*/   OPC_EmitInteger, MVT::i32, 1, 
/* 32900*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 32903*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32906*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32909*/   OPC_MorphNodeTo1, TARGET_VAL(R600::MULHI_INT_r600), 0,
                 MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
             // Src: (mulhs:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
             // Dst: (MULHI_INT_r600:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 32935*/  /*Scope*/ 100, /*->33036*/
/* 32936*/   OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 32938*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32941*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32944*/   OPC_EmitInteger, MVT::i32, 1, 
/* 32947*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32950*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32953*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32956*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32959*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32962*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32965*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 32977*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32980*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32983*/   OPC_EmitInteger, MVT::i32, 0, 
/* 32986*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 32998*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33001*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 33004*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33007*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33010*/   OPC_MorphNodeTo1, TARGET_VAL(R600::MULHI_INT_eg), 0,
                 MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
             // Src: (mulhs:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
             // Dst: (MULHI_INT_eg:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 33036*/  /*Scope*/ 100, /*->33137*/
/* 33037*/   OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 33039*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33042*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33045*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33048*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33051*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33054*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33057*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33060*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33063*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33066*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 33078*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33081*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33084*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33087*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 33099*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33102*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 33105*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33108*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33111*/   OPC_MorphNodeTo1, TARGET_VAL(R600::MULHI_INT_cm), 0,
                 MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
             // Src: (mulhs:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
             // Dst: (MULHI_INT_cm:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 33137*/  0, /*End of Scope*/
/* 33138*/ /*SwitchOpcode*/ 53|128,2/*309*/, TARGET_VAL(ISD::MULHU),// ->33451
/* 33142*/  OPC_RecordChild0, // #0 = $src0
/* 33143*/  OPC_RecordChild1, // #1 = $src1
/* 33144*/  OPC_CheckType, MVT::i32,
/* 33146*/  OPC_Scope, 100, /*->33248*/ // 3 children in Scope
/* 33148*/   OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 33150*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33153*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33156*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33159*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33162*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33165*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33168*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33171*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33174*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33177*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 33189*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33192*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33195*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33198*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 33210*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33213*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 33216*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33219*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33222*/   OPC_MorphNodeTo1, TARGET_VAL(R600::MULHI_UINT_r600), 0,
                 MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
             // Src: (mulhu:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
             // Dst: (MULHI_UINT_r600:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 33248*/  /*Scope*/ 100, /*->33349*/
/* 33249*/   OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 33251*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33254*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33257*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33260*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33263*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33266*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33269*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33272*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33275*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33278*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 33290*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33293*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33296*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33299*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 33311*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33314*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 33317*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33320*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33323*/   OPC_MorphNodeTo1, TARGET_VAL(R600::MULHI_UINT_eg), 0,
                 MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
             // Src: (mulhu:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
             // Dst: (MULHI_UINT_eg:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 33349*/  /*Scope*/ 100, /*->33450*/
/* 33350*/   OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 33352*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33355*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33358*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33361*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33364*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33367*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33370*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33373*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33376*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33379*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 33391*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33394*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33397*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33400*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 33412*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33415*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 33418*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33421*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33424*/   OPC_MorphNodeTo1, TARGET_VAL(R600::MULHI_UINT_cm), 0,
                 MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
             // Src: (mulhu:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
             // Dst: (MULHI_UINT_cm:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 33450*/  0, /*End of Scope*/
/* 33451*/ /*SwitchOpcode*/ 63|128,3/*447*/, TARGET_VAL(AMDGPUISD::URECIP),// ->33902
/* 33455*/  OPC_RecordChild0, // #0 = $src0
/* 33456*/  OPC_CheckType, MVT::i32,
/* 33458*/  OPC_Scope, 66, /*->33526*/ // 3 children in Scope
/* 33460*/   OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 33462*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33465*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33468*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33471*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33474*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33477*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33480*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33483*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 33495*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33498*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 33501*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33504*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33507*/   OPC_MorphNodeTo1, TARGET_VAL(R600::RECIP_UINT_r600), 0,
                 MVT::i32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (AMDGPUurecip:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0) - Complexity = 3
             // Dst: (RECIP_UINT_r600:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0)
/* 33526*/  /*Scope*/ 66, /*->33593*/
/* 33527*/   OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 33529*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33532*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33535*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33538*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33541*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33544*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33547*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33550*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 33562*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33565*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 33568*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33571*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33574*/   OPC_MorphNodeTo1, TARGET_VAL(R600::RECIP_UINT_eg), 0,
                 MVT::i32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (AMDGPUurecip:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0) - Complexity = 3
             // Dst: (RECIP_UINT_eg:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0)
/* 33593*/  /*Scope*/ 50|128,2/*306*/, /*->33901*/
/* 33595*/   OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 33597*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33600*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33603*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33606*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33609*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33612*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33615*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33618*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33621*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33624*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33627*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33630*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33633*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33636*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33639*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33642*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33645*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33648*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33651*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33654*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33657*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33660*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 33672*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33675*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 33678*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33681*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33684*/   OPC_EmitNode1, TARGET_VAL(R600::UINT_TO_FLT_eg), 0,
                 MVT::i32, 13/*#Ops*/, 15, 16, 17, 18, 0, 19, 20, 21, 22, 23, 24, 25, 26,  // Results = #27
/* 33703*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33706*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33709*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33712*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 33724*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33727*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 33730*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33733*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33736*/   OPC_EmitNode1, TARGET_VAL(R600::RECIP_IEEE_cm), 0,
                 MVT::i32, 13/*#Ops*/, 11, 12, 13, 14, 27, 28, 29, 30, 31, 32, 33, 34, 35,  // Results = #36
/* 33755*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33758*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33761*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33764*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 33776*/   OPC_EmitInteger, MVT::i32, 126|128,127|128,127|128,123|128,4/*1333788670*/, 
/* 33783*/   OPC_EmitNode1, TARGET_VAL(R600::MOV_IMM_I32), 0,
                 MVT::i32, 1/*#Ops*/, 41,  // Results = #42
/* 33790*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33793*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33796*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33799*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 33811*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33814*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 33817*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33820*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33823*/   OPC_EmitNode1, TARGET_VAL(R600::MUL_IEEE), 0,
                 MVT::i32, 20/*#Ops*/, 5, 6, 7, 8, 9, 10, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50,  // Results = #51
/* 33849*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33852*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33855*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33858*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 33870*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33873*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 33876*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33879*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33882*/   OPC_MorphNodeTo1, TARGET_VAL(R600::FLT_TO_UINT_eg), 0,
                 MVT::i32, 13/*#Ops*/, 1, 2, 3, 4, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
             // Src: (AMDGPUurecip:{ *:[i32] } i32:{ *:[i32] }:$src0) - Complexity = 3
             // Dst: (FLT_TO_UINT_eg:{ *:[i32] } (MUL_IEEE:{ *:[i32] } (RECIP_IEEE_cm:{ *:[i32] } (UINT_TO_FLT_eg:{ *:[i32] } ?:{ *:[i32] }:$src0)), (MOV_IMM_I32:{ *:[i32] } 1333788670:{ *:[i32] })))
/* 33901*/  0, /*End of Scope*/
/* 33902*/ /*SwitchOpcode*/ 80|128,1/*208*/, TARGET_VAL(AMDGPUISD::MULHI_U24),// ->34114
/* 33906*/  OPC_RecordChild0, // #0 = $src0
/* 33907*/  OPC_RecordChild1, // #1 = $src1
/* 33908*/  OPC_CheckType, MVT::i32,
/* 33910*/  OPC_Scope, 100, /*->34012*/ // 2 children in Scope
/* 33912*/   OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 33914*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33917*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33920*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33923*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33926*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33929*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33932*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33935*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33938*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33941*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 33953*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33956*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33959*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33962*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 33974*/   OPC_EmitInteger, MVT::i32, 1, 
/* 33977*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 33980*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33983*/   OPC_EmitInteger, MVT::i32, 0, 
/* 33986*/   OPC_MorphNodeTo1, TARGET_VAL(R600::MULHI_UINT24_eg), 0,
                 MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
             // Src: (AMDGPUmulhi_u24:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
             // Dst: (MULHI_UINT24_eg:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 34012*/  /*Scope*/ 100, /*->34113*/
/* 34013*/   OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 34015*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34018*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34021*/   OPC_EmitInteger, MVT::i32, 1, 
/* 34024*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34027*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34030*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34033*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34036*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34039*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34042*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34054*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34057*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34060*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34063*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34075*/   OPC_EmitInteger, MVT::i32, 1, 
/* 34078*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 34081*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34084*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34087*/   OPC_MorphNodeTo1, TARGET_VAL(R600::MULHI_UINT_cm24), 0,
                 MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
             // Src: (AMDGPUmulhi_u24:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
             // Dst: (MULHI_UINT_cm24:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 34113*/  0, /*End of Scope*/
/* 34114*/ /*SwitchOpcode*/ 105, TARGET_VAL(AMDGPUISD::BFE_U32),// ->34222
/* 34117*/  OPC_RecordChild0, // #0 = $src0
/* 34118*/  OPC_RecordChild1, // #1 = $src1
/* 34119*/  OPC_RecordChild2, // #2 = $src2
/* 34120*/  OPC_CheckChild2Type, MVT::i32,
/* 34122*/  OPC_CheckType, MVT::i32,
/* 34124*/  OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 34126*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34129*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34132*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34135*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34138*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34150*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34153*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34156*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34168*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34171*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34174*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34186*/  OPC_EmitInteger, MVT::i32, 1, 
/* 34189*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 34192*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34195*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34198*/  OPC_MorphNodeTo1, TARGET_VAL(R600::BFE_UINT_eg), 0,
                MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
            // Src: (AMDGPUbfe_u32_impl:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2) - Complexity = 3
            // Dst: (BFE_UINT_eg:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2)
/* 34222*/ /*SwitchOpcode*/ 105, TARGET_VAL(AMDGPUISD::BFE_I32),// ->34330
/* 34225*/  OPC_RecordChild0, // #0 = $src0
/* 34226*/  OPC_RecordChild1, // #1 = $src1
/* 34227*/  OPC_RecordChild2, // #2 = $src2
/* 34228*/  OPC_CheckChild2Type, MVT::i32,
/* 34230*/  OPC_CheckType, MVT::i32,
/* 34232*/  OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 34234*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34237*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34240*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34243*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34246*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34258*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34261*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34264*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34276*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34279*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34282*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34294*/  OPC_EmitInteger, MVT::i32, 1, 
/* 34297*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 34300*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34303*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34306*/  OPC_MorphNodeTo1, TARGET_VAL(R600::BFE_INT_eg), 0,
                MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
            // Src: (AMDGPUbfe_i32_impl:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2) - Complexity = 3
            // Dst: (BFE_INT_eg:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2)
/* 34330*/ /*SwitchOpcode*/ 105, TARGET_VAL(AMDGPUISD::BFI),// ->34438
/* 34333*/  OPC_RecordChild0, // #0 = $src0
/* 34334*/  OPC_RecordChild1, // #1 = $src1
/* 34335*/  OPC_RecordChild2, // #2 = $src2
/* 34336*/  OPC_CheckChild2Type, MVT::i32,
/* 34338*/  OPC_CheckType, MVT::i32,
/* 34340*/  OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 34342*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34345*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34348*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34351*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34354*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34366*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34369*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34372*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34384*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34387*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34390*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34402*/  OPC_EmitInteger, MVT::i32, 1, 
/* 34405*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 34408*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34411*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34414*/  OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
            // Src: (AMDGPUbfi:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2) - Complexity = 3
            // Dst: (BFI_INT_eg:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2)
/* 34438*/ /*SwitchOpcode*/ 92|128,2/*348*/, TARGET_VAL(ISD::SIGN_EXTEND_INREG),// ->34790
/* 34442*/  OPC_RecordChild0, // #0 = $src
/* 34443*/  OPC_MoveChild1,
/* 34444*/  OPC_Scope, 109, /*->34555*/ // 3 children in Scope
/* 34446*/   OPC_CheckValueType, MVT::i1,
/* 34448*/   OPC_MoveParent,
/* 34449*/   OPC_CheckType, MVT::i32,
/* 34451*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 34453*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34456*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34459*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34462*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34465*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34477*/   OPC_EmitRegister, MVT::i32, R600::ZERO,
/* 34480*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34483*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34486*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34498*/   OPC_EmitRegister, MVT::i32, R600::ONE_INT,
/* 34501*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34504*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34507*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34519*/   OPC_EmitInteger, MVT::i32, 1, 
/* 34522*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 34525*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34528*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34531*/   OPC_MorphNodeTo1, TARGET_VAL(R600::BFE_INT_eg), 0,
                 MVT::i32, 18/*#Ops*/, 1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 
             // Src: (sext_inreg:{ *:[i32] } i32:{ *:[i32] }:$src, i1:{ *:[Other] }) - Complexity = 3
             // Dst: (BFE_INT_eg:{ *:[i32] } i32:{ *:[i32] }:$src, ZERO:{ *:[i32] }, ONE_INT:{ *:[i32] })
/* 34555*/  /*Scope*/ 116, /*->34672*/
/* 34556*/   OPC_CheckValueType, MVT::i8,
/* 34558*/   OPC_MoveParent,
/* 34559*/   OPC_CheckType, MVT::i32,
/* 34561*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 34563*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34566*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34569*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34572*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34575*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34587*/   OPC_EmitRegister, MVT::i32, R600::ZERO,
/* 34590*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34593*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34596*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34608*/   OPC_EmitInteger, MVT::i32, 8, 
/* 34611*/   OPC_EmitNode1, TARGET_VAL(R600::MOV_IMM_I32), 0,
                 MVT::i32, 1/*#Ops*/, 10,  // Results = #11
/* 34618*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34621*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34624*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34636*/   OPC_EmitInteger, MVT::i32, 1, 
/* 34639*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 34642*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34645*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34648*/   OPC_MorphNodeTo1, TARGET_VAL(R600::BFE_INT_eg), 0,
                 MVT::i32, 18/*#Ops*/, 1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 
             // Src: (sext_inreg:{ *:[i32] } i32:{ *:[i32] }:$src, i8:{ *:[Other] }) - Complexity = 3
             // Dst: (BFE_INT_eg:{ *:[i32] } i32:{ *:[i32] }:$src, ZERO:{ *:[i32] }, (MOV_IMM_I32:{ *:[i32] } 8:{ *:[i32] }))
/* 34672*/  /*Scope*/ 116, /*->34789*/
/* 34673*/   OPC_CheckValueType, MVT::i16,
/* 34675*/   OPC_MoveParent,
/* 34676*/   OPC_CheckType, MVT::i32,
/* 34678*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 34680*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34683*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34686*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34689*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34692*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34704*/   OPC_EmitRegister, MVT::i32, R600::ZERO,
/* 34707*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34710*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34713*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34725*/   OPC_EmitInteger, MVT::i32, 16, 
/* 34728*/   OPC_EmitNode1, TARGET_VAL(R600::MOV_IMM_I32), 0,
                 MVT::i32, 1/*#Ops*/, 10,  // Results = #11
/* 34735*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34738*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34741*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34753*/   OPC_EmitInteger, MVT::i32, 1, 
/* 34756*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 34759*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34762*/   OPC_EmitInteger, MVT::i32, 0, 
/* 34765*/   OPC_MorphNodeTo1, TARGET_VAL(R600::BFE_INT_eg), 0,
                 MVT::i32, 18/*#Ops*/, 1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 
             // Src: (sext_inreg:{ *:[i32] } i32:{ *:[i32] }:$src, i16:{ *:[Other] }) - Complexity = 3
             // Dst: (BFE_INT_eg:{ *:[i32] } i32:{ *:[i32] }:$src, ZERO:{ *:[i32] }, (MOV_IMM_I32:{ *:[i32] } 16:{ *:[i32] }))
/* 34789*/  0, /*End of Scope*/
/* 34790*/ /*SwitchOpcode*/ 104, TARGET_VAL(AMDGPUISD::BFM),// ->34897
/* 34793*/  OPC_RecordChild0, // #0 = $src0
/* 34794*/  OPC_RecordChild1, // #1 = $src1
/* 34795*/  OPC_CheckType, MVT::i32,
/* 34797*/  OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 34799*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34802*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34805*/  OPC_EmitInteger, MVT::i32, 1, 
/* 34808*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34811*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34814*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34817*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34820*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34823*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34826*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34838*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34841*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34844*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34847*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34859*/  OPC_EmitInteger, MVT::i32, 1, 
/* 34862*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 34865*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34868*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34871*/  OPC_MorphNodeTo1, TARGET_VAL(R600::BFM_INT_eg), 0,
                MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
            // Src: (AMDGPUbfm:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1) - Complexity = 3
            // Dst: (BFM_INT_eg:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 34897*/ /*SwitchOpcode*/ 105, TARGET_VAL(AMDGPUISD::MAD_U24),// ->35005
/* 34900*/  OPC_RecordChild0, // #0 = $src0
/* 34901*/  OPC_RecordChild1, // #1 = $src1
/* 34902*/  OPC_RecordChild2, // #2 = $src2
/* 34903*/  OPC_CheckChild2Type, MVT::i32,
/* 34905*/  OPC_CheckType, MVT::i32,
/* 34907*/  OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 34909*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34912*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34915*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34918*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34921*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34933*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34936*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34939*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34951*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34954*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34957*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 34969*/  OPC_EmitInteger, MVT::i32, 1, 
/* 34972*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 34975*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34978*/  OPC_EmitInteger, MVT::i32, 0, 
/* 34981*/  OPC_MorphNodeTo1, TARGET_VAL(R600::MULADD_UINT24_eg), 0,
                MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
            // Src: (AMDGPUmad_u24:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2) - Complexity = 3
            // Dst: (MULADD_UINT24_eg:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2)
/* 35005*/ /*SwitchOpcode*/ 105, TARGET_VAL(ISD::FSHR),// ->35113
/* 35008*/  OPC_RecordChild0, // #0 = $src0
/* 35009*/  OPC_RecordChild1, // #1 = $src1
/* 35010*/  OPC_RecordChild2, // #2 = $src2
/* 35011*/  OPC_CheckChild2Type, MVT::i32,
/* 35013*/  OPC_CheckType, MVT::i32,
/* 35015*/  OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 35017*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35020*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35023*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35026*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35029*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 35041*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35044*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35047*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 35059*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35062*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35065*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 35077*/  OPC_EmitInteger, MVT::i32, 1, 
/* 35080*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 35083*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35086*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35089*/  OPC_MorphNodeTo1, TARGET_VAL(R600::BIT_ALIGN_INT_eg), 0,
                MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
            // Src: (fshr:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2) - Complexity = 3
            // Dst: (BIT_ALIGN_INT_eg:{ *:[i32] } ?:{ *:[i32] }:$src0, ?:{ *:[i32] }:$src1, ?:{ *:[i32] }:$src2)
/* 35113*/ /*SwitchOpcode*/ 104, TARGET_VAL(ISD::ROTR),// ->35220
/* 35116*/  OPC_RecordChild0, // #0 = $src0
/* 35117*/  OPC_RecordChild1, // #1 = $src1
/* 35118*/  OPC_CheckChild1Type, MVT::i32,
/* 35120*/  OPC_CheckType, MVT::i32,
/* 35122*/  OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 35124*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35127*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35130*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35133*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35136*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 35148*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35151*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35154*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 35166*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35169*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35172*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 35184*/  OPC_EmitInteger, MVT::i32, 1, 
/* 35187*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 35190*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35193*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35196*/  OPC_MorphNodeTo1, TARGET_VAL(R600::BIT_ALIGN_INT_eg), 0,
                MVT::i32, 18/*#Ops*/, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 1, 10, 11, 12, 13, 14, 15, 16, 
            // Src: (rotr:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1) - Complexity = 3
            // Dst: (BIT_ALIGN_INT_eg:{ *:[i32] } ?:{ *:[i32] }:$src0, ?:{ *:[i32] }:$src0, ?:{ *:[i32] }:$src1)
/* 35220*/ /*SwitchOpcode*/ 104, TARGET_VAL(AMDGPUISD::MUL_U24),// ->35327
/* 35223*/  OPC_RecordChild0, // #0 = $src0
/* 35224*/  OPC_RecordChild1, // #1 = $src1
/* 35225*/  OPC_CheckType, MVT::i32,
/* 35227*/  OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 35229*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35232*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35235*/  OPC_EmitInteger, MVT::i32, 1, 
/* 35238*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35241*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35244*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35247*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35250*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35253*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35256*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 35268*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35271*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35274*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35277*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 35289*/  OPC_EmitInteger, MVT::i32, 1, 
/* 35292*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 35295*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35298*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35301*/  OPC_MorphNodeTo1, TARGET_VAL(R600::MUL_UINT24_eg), 0,
                MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
            // Src: (AMDGPUmul_u24_impl:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1) - Complexity = 3
            // Dst: (MUL_UINT24_eg:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 35327*/ /*SwitchOpcode*/ 104, TARGET_VAL(AMDGPUISD::CARRY),// ->35434
/* 35330*/  OPC_RecordChild0, // #0 = $src0
/* 35331*/  OPC_RecordChild1, // #1 = $src1
/* 35332*/  OPC_CheckType, MVT::i32,
/* 35334*/  OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 35336*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35339*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35342*/  OPC_EmitInteger, MVT::i32, 1, 
/* 35345*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35348*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35351*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35354*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35357*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35360*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35363*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 35375*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35378*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35381*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35384*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 35396*/  OPC_EmitInteger, MVT::i32, 1, 
/* 35399*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 35402*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35405*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35408*/  OPC_MorphNodeTo1, TARGET_VAL(R600::ADDC_UINT), 0,
                MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
            // Src: (AMDGPUcarry:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
            // Dst: (ADDC_UINT:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 35434*/ /*SwitchOpcode*/ 104, TARGET_VAL(AMDGPUISD::BORROW),// ->35541
/* 35437*/  OPC_RecordChild0, // #0 = $src0
/* 35438*/  OPC_RecordChild1, // #1 = $src1
/* 35439*/  OPC_CheckType, MVT::i32,
/* 35441*/  OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 35443*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35446*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35449*/  OPC_EmitInteger, MVT::i32, 1, 
/* 35452*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35455*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35458*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35461*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35464*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35467*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35470*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 35482*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35485*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35488*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35491*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 35503*/  OPC_EmitInteger, MVT::i32, 1, 
/* 35506*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 35509*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35512*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35515*/  OPC_MorphNodeTo1, TARGET_VAL(R600::SUBB_UINT), 0,
                MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
            // Src: (AMDGPUborrow:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
            // Dst: (SUBB_UINT:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 35541*/ /*SwitchOpcode*/ 71, TARGET_VAL(AMDGPUISD::FP_TO_FP16),// ->35615
/* 35544*/  OPC_RecordChild0, // #0 = $src0
/* 35545*/  OPC_CheckChild0Type, MVT::f32,
/* 35547*/  OPC_CheckType, MVT::i32,
/* 35549*/  OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 35551*/  OPC_EmitInteger, MVT::i32, 1, 
/* 35554*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35557*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35560*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35563*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35566*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35569*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35572*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 35584*/  OPC_EmitInteger, MVT::i32, 1, 
/* 35587*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 35590*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35593*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35596*/  OPC_MorphNodeTo1, TARGET_VAL(R600::FLT32_TO_FLT16), 0,
                MVT::i32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
            // Src: (AMDGPUfp_to_f16:{ *:[i32] } R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
            // Dst: (FLT32_TO_FLT16:{ *:[i32] } R600_Reg32:{ *:[f32] }:$src0)
/* 35615*/ /*SwitchOpcode*/ 71, TARGET_VAL(ISD::CTPOP),// ->35689
/* 35618*/  OPC_RecordChild0, // #0 = $src0
/* 35619*/  OPC_CheckChild0Type, MVT::i32,
/* 35621*/  OPC_CheckType, MVT::i32,
/* 35623*/  OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 35625*/  OPC_EmitInteger, MVT::i32, 1, 
/* 35628*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35631*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35634*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35637*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35640*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35643*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35646*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 35658*/  OPC_EmitInteger, MVT::i32, 1, 
/* 35661*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 35664*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35667*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35670*/  OPC_MorphNodeTo1, TARGET_VAL(R600::BCNT_INT), 0,
                MVT::i32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
            // Src: (ctpop:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0) - Complexity = 3
            // Dst: (BCNT_INT:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0)
/* 35689*/ /*SwitchOpcode*/ 71, TARGET_VAL(ISD::CTLZ_ZERO_UNDEF),// ->35763
/* 35692*/  OPC_RecordChild0, // #0 = $src0
/* 35693*/  OPC_CheckChild0Type, MVT::i32,
/* 35695*/  OPC_CheckType, MVT::i32,
/* 35697*/  OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 35699*/  OPC_EmitInteger, MVT::i32, 1, 
/* 35702*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35705*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35708*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35711*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35714*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35717*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35720*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 35732*/  OPC_EmitInteger, MVT::i32, 1, 
/* 35735*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 35738*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35741*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35744*/  OPC_MorphNodeTo1, TARGET_VAL(R600::FFBH_UINT), 0,
                MVT::i32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
            // Src: (ctlz_zero_undef:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0) - Complexity = 3
            // Dst: (FFBH_UINT:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0)
/* 35763*/ /*SwitchOpcode*/ 71, TARGET_VAL(AMDGPUISD::FFBH_U32),// ->35837
/* 35766*/  OPC_RecordChild0, // #0 = $src0
/* 35767*/  OPC_CheckChild0Type, MVT::i32,
/* 35769*/  OPC_CheckType, MVT::i32,
/* 35771*/  OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 35773*/  OPC_EmitInteger, MVT::i32, 1, 
/* 35776*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35779*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35782*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35785*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35788*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35791*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35794*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 35806*/  OPC_EmitInteger, MVT::i32, 1, 
/* 35809*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 35812*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35815*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35818*/  OPC_MorphNodeTo1, TARGET_VAL(R600::FFBH_UINT), 0,
                MVT::i32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
            // Src: (AMDGPUffbh_u32_impl:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0) - Complexity = 3
            // Dst: (FFBH_UINT:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0)
/* 35837*/ /*SwitchOpcode*/ 71, TARGET_VAL(ISD::CTTZ_ZERO_UNDEF),// ->35911
/* 35840*/  OPC_RecordChild0, // #0 = $src0
/* 35841*/  OPC_CheckChild0Type, MVT::i32,
/* 35843*/  OPC_CheckType, MVT::i32,
/* 35845*/  OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 35847*/  OPC_EmitInteger, MVT::i32, 1, 
/* 35850*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35853*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35856*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35859*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35862*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35865*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35868*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 35880*/  OPC_EmitInteger, MVT::i32, 1, 
/* 35883*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 35886*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35889*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35892*/  OPC_MorphNodeTo1, TARGET_VAL(R600::FFBL_INT), 0,
                MVT::i32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
            // Src: (cttz_zero_undef:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0) - Complexity = 3
            // Dst: (FFBL_INT:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0)
/* 35911*/ /*SwitchOpcode*/ 71, TARGET_VAL(AMDGPUISD::FFBL_B32),// ->35985
/* 35914*/  OPC_RecordChild0, // #0 = $src0
/* 35915*/  OPC_CheckChild0Type, MVT::i32,
/* 35917*/  OPC_CheckType, MVT::i32,
/* 35919*/  OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 35921*/  OPC_EmitInteger, MVT::i32, 1, 
/* 35924*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35927*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35930*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35933*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35936*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35939*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35942*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 35954*/  OPC_EmitInteger, MVT::i32, 1, 
/* 35957*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 35960*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35963*/  OPC_EmitInteger, MVT::i32, 0, 
/* 35966*/  OPC_MorphNodeTo1, TARGET_VAL(R600::FFBL_INT), 0,
                MVT::i32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
            // Src: (AMDGPUffbl_b32_impl:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0) - Complexity = 3
            // Dst: (FFBL_INT:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0)
/* 35985*/ /*SwitchOpcode*/ 105, TARGET_VAL(AMDGPUISD::MAD_I24),// ->36093
/* 35988*/  OPC_RecordChild0, // #0 = $src0
/* 35989*/  OPC_RecordChild1, // #1 = $src1
/* 35990*/  OPC_RecordChild2, // #2 = $src2
/* 35991*/  OPC_CheckChild2Type, MVT::i32,
/* 35993*/  OPC_CheckType, MVT::i32,
/* 35995*/  OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 35997*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36000*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36003*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36006*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36009*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 36021*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36024*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36027*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 36039*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36042*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36045*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 36057*/  OPC_EmitInteger, MVT::i32, 1, 
/* 36060*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 36063*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36066*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36069*/  OPC_MorphNodeTo1, TARGET_VAL(R600::MULADD_INT24_cm), 0,
                MVT::i32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
            // Src: (AMDGPUmad_i24:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2) - Complexity = 3
            // Dst: (MULADD_INT24_cm:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2)
/* 36093*/ /*SwitchOpcode*/ 104, TARGET_VAL(AMDGPUISD::MUL_I24),// ->36200
/* 36096*/  OPC_RecordChild0, // #0 = $src0
/* 36097*/  OPC_RecordChild1, // #1 = $src1
/* 36098*/  OPC_CheckType, MVT::i32,
/* 36100*/  OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 36102*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36105*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36108*/  OPC_EmitInteger, MVT::i32, 1, 
/* 36111*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36114*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36117*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36120*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36123*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36126*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36129*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 36141*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36144*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36147*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36150*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 36162*/  OPC_EmitInteger, MVT::i32, 1, 
/* 36165*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 36168*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36171*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36174*/  OPC_MorphNodeTo1, TARGET_VAL(R600::MUL_INT24_cm), 0,
                MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
            // Src: (AMDGPUmul_i24_impl:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1) - Complexity = 3
            // Dst: (MUL_INT24_cm:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)
/* 36200*/ /*SwitchOpcode*/ 104, TARGET_VAL(AMDGPUISD::MULHI_I24),// ->36307
/* 36203*/  OPC_RecordChild0, // #0 = $src0
/* 36204*/  OPC_RecordChild1, // #1 = $src1
/* 36205*/  OPC_CheckType, MVT::i32,
/* 36207*/  OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 36209*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36212*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36215*/  OPC_EmitInteger, MVT::i32, 1, 
/* 36218*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36221*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36224*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36227*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36230*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36233*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36236*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 36248*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36251*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36254*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36257*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 36269*/  OPC_EmitInteger, MVT::i32, 1, 
/* 36272*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 36275*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36278*/  OPC_EmitInteger, MVT::i32, 0, 
/* 36281*/  OPC_MorphNodeTo1, TARGET_VAL(R600::MULHI_INT_cm24), 0,
                MVT::i32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
            // Src: (AMDGPUmulhi_i24:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1) - Complexity = 3
            // Dst: (MULHI_INT_cm24:{ *:[i32] } R600_Reg32:{ *:[i32] }:$src0, R600_Reg32:{ *:[i32] }:$src1)
/* 36307*/ /*SwitchOpcode*/ 8, TARGET_VAL(AMDGPUISD::ENDPGM),// ->36318
/* 36310*/  OPC_RecordNode, // #0 = 'AMDGPUendpgm' chained node
/* 36311*/  OPC_CaptureGlueInput,
/* 36312*/  OPC_EmitMergeInputChains1_0,
/* 36313*/  OPC_MorphNodeTo0, TARGET_VAL(R600::RETURN), 0|OPFL_Chain|OPFL_GlueInput,
                0/*#Ops*/, 
            // Src: (AMDGPUendpgm) - Complexity = 3
            // Dst: (RETURN)
/* 36318*/ /*SwitchOpcode*/ 14, TARGET_VAL(ISD::Constant),// ->36335
/* 36321*/  OPC_RecordNode, // #0 = $val
/* 36322*/  OPC_CheckType, MVT::i32,
/* 36324*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 36326*/  OPC_EmitConvertToTarget, 0,
/* 36328*/  OPC_MorphNodeTo1, TARGET_VAL(R600::MOV_IMM_I32), 0,
                MVT::i32, 1/*#Ops*/, 1, 
            // Src: (imm:{ *:[i32] }):$val - Complexity = 3
            // Dst: (MOV_IMM_I32:{ *:[i32] } (imm:{ *:[i32] }):$val)
/* 36335*/ /*SwitchOpcode*/ 14, TARGET_VAL(ISD::BR),// ->36352
/* 36338*/  OPC_RecordNode, // #0 = 'br' chained node
/* 36339*/  OPC_RecordChild1, // #1 = $target
/* 36340*/  OPC_MoveChild1,
/* 36341*/  OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/* 36344*/  OPC_MoveParent,
/* 36345*/  OPC_EmitMergeInputChains1_0,
/* 36346*/  OPC_MorphNodeTo0, TARGET_VAL(R600::BRANCH), 0|OPFL_Chain,
                1/*#Ops*/, 1, 
            // Src: (br (bb:{ *:[Other] }):$target) - Complexity = 3
            // Dst: (BRANCH (bb:{ *:[Other] }):$target)
/* 36352*/ /*SwitchOpcode*/ 32, TARGET_VAL(AMDGPUISD::BRANCH_COND),// ->36387
/* 36355*/  OPC_RecordNode, // #0 = 'IL_brcond' chained node
/* 36356*/  OPC_RecordChild1, // #1 = $target
/* 36357*/  OPC_MoveChild1,
/* 36358*/  OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/* 36361*/  OPC_MoveParent,
/* 36362*/  OPC_RecordChild2, // #2 = $src0
/* 36363*/  OPC_Scope, 10, /*->36375*/ // 2 children in Scope
/* 36365*/   OPC_CheckChild2Type, MVT::i32,
/* 36367*/   OPC_EmitMergeInputChains1_0,
/* 36368*/   OPC_MorphNodeTo0, TARGET_VAL(R600::BRANCH_COND_i32), 0|OPFL_Chain,
                 2/*#Ops*/, 1, 2, 
             // Src: (IL_brcond (bb:{ *:[Other] }):$target, R600_Reg32:{ *:[i32] }:$src0) - Complexity = 3
             // Dst: (BRANCH_COND_i32 (bb:{ *:[Other] }):$target, R600_Reg32:{ *:[i32] }:$src0)
/* 36375*/  /*Scope*/ 10, /*->36386*/
/* 36376*/   OPC_CheckChild2Type, MVT::f32,
/* 36378*/   OPC_EmitMergeInputChains1_0,
/* 36379*/   OPC_MorphNodeTo0, TARGET_VAL(R600::BRANCH_COND_f32), 0|OPFL_Chain,
                 2/*#Ops*/, 1, 2, 
             // Src: (IL_brcond (bb:{ *:[Other] }):$target, R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (BRANCH_COND_f32 (bb:{ *:[Other] }):$target, R600_Reg32:{ *:[f32] }:$src0)
/* 36386*/  0, /*End of Scope*/
/* 36387*/ /*SwitchOpcode*/ 83|128,5/*723*/, TARGET_VAL(ISD::FDIV),// ->37114
/* 36391*/  OPC_Scope, 85|128,1/*213*/, /*->36607*/ // 2 children in Scope
/* 36394*/   OPC_MoveChild0,
/* 36395*/   OPC_CheckOpcode, TARGET_VAL(ISD::ConstantFP),
/* 36398*/   OPC_CheckPredicate, 2, // Predicate_FP_ONE
/* 36400*/   OPC_MoveParent,
/* 36401*/   OPC_RecordChild1, // #0 = $src
/* 36402*/   OPC_CheckType, MVT::f32,
/* 36404*/   OPC_Scope, 66, /*->36472*/ // 3 children in Scope
/* 36406*/    OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 36408*/    OPC_EmitInteger, MVT::i32, 1, 
/* 36411*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36414*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36417*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36420*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36423*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36426*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36429*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 36441*/    OPC_EmitInteger, MVT::i32, 1, 
/* 36444*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 36447*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36450*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36453*/    OPC_MorphNodeTo1, TARGET_VAL(R600::RECIP_IEEE_r600), 0,
                  MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
              // Src: (fdiv:{ *:[f32] } (fpimm:{ *:[f32] })<<P:Predicate_FP_ONE>>, f32:{ *:[f32] }:$src) - Complexity = 7
              // Dst: (RECIP_IEEE_r600:{ *:[f32] } ?:{ *:[f32] }:$src)
/* 36472*/   /*Scope*/ 66, /*->36539*/
/* 36473*/    OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 36475*/    OPC_EmitInteger, MVT::i32, 1, 
/* 36478*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36481*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36484*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36487*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36490*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36493*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36496*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 36508*/    OPC_EmitInteger, MVT::i32, 1, 
/* 36511*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 36514*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36517*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36520*/    OPC_MorphNodeTo1, TARGET_VAL(R600::RECIP_IEEE_eg), 0,
                  MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
              // Src: (fdiv:{ *:[f32] } (fpimm:{ *:[f32] })<<P:Predicate_FP_ONE>>, f32:{ *:[f32] }:$src) - Complexity = 7
              // Dst: (RECIP_IEEE_eg:{ *:[f32] } ?:{ *:[f32] }:$src)
/* 36539*/   /*Scope*/ 66, /*->36606*/
/* 36540*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 36542*/    OPC_EmitInteger, MVT::i32, 1, 
/* 36545*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36548*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36551*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36554*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36557*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36560*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36563*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 36575*/    OPC_EmitInteger, MVT::i32, 1, 
/* 36578*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 36581*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36584*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36587*/    OPC_MorphNodeTo1, TARGET_VAL(R600::RECIP_IEEE_cm), 0,
                  MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
              // Src: (fdiv:{ *:[f32] } (fpimm:{ *:[f32] })<<P:Predicate_FP_ONE>>, f32:{ *:[f32] }:$src) - Complexity = 7
              // Dst: (RECIP_IEEE_cm:{ *:[f32] } ?:{ *:[f32] }:$src)
/* 36606*/   0, /*End of Scope*/
/* 36607*/  /*Scope*/ 120|128,3/*504*/, /*->37113*/
/* 36609*/   OPC_RecordChild0, // #0 = $src0
/* 36610*/   OPC_RecordChild1, // #1 = $src1
/* 36611*/   OPC_CheckType, MVT::f32,
/* 36613*/   OPC_Scope, 36|128,1/*164*/, /*->36780*/ // 3 children in Scope
/* 36616*/    OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 36618*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36621*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36624*/    OPC_EmitInteger, MVT::i32, 1, 
/* 36627*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36630*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36633*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36636*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36639*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36642*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36645*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 36657*/    OPC_EmitInteger, MVT::i32, 1, 
/* 36660*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36663*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36666*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36669*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36672*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36675*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36678*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 36690*/    OPC_EmitInteger, MVT::i32, 1, 
/* 36693*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 36696*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36699*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36702*/    OPC_EmitNode1, TARGET_VAL(R600::RECIP_IEEE_r600), 0,
                  MVT::i32, 13/*#Ops*/, 12, 13, 14, 15, 1, 16, 17, 18, 19, 20, 21, 22, 23,  // Results = #24
/* 36721*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36724*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36727*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36730*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 36742*/    OPC_EmitInteger, MVT::i32, 1, 
/* 36745*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 36748*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36751*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36754*/    OPC_MorphNodeTo1, TARGET_VAL(R600::MUL_IEEE), 0,
                  MVT::f32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 24, 25, 26, 27, 28, 29, 30, 31, 32, 
              // Src: (fdiv:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1) - Complexity = 3
              // Dst: (MUL_IEEE:{ *:[f32] } ?:{ *:[f32] }:$src0, (RECIP_IEEE_r600:{ *:[i32] } ?:{ *:[f32] }:$src1))
/* 36780*/   /*Scope*/ 36|128,1/*164*/, /*->36946*/
/* 36782*/    OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 36784*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36787*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36790*/    OPC_EmitInteger, MVT::i32, 1, 
/* 36793*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36796*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36799*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36802*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36805*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36808*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36811*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 36823*/    OPC_EmitInteger, MVT::i32, 1, 
/* 36826*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36829*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36832*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36835*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36838*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36841*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36844*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 36856*/    OPC_EmitInteger, MVT::i32, 1, 
/* 36859*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 36862*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36865*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36868*/    OPC_EmitNode1, TARGET_VAL(R600::RECIP_IEEE_eg), 0,
                  MVT::i32, 13/*#Ops*/, 12, 13, 14, 15, 1, 16, 17, 18, 19, 20, 21, 22, 23,  // Results = #24
/* 36887*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36890*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36893*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36896*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 36908*/    OPC_EmitInteger, MVT::i32, 1, 
/* 36911*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 36914*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36917*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36920*/    OPC_MorphNodeTo1, TARGET_VAL(R600::MUL_IEEE), 0,
                  MVT::f32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 24, 25, 26, 27, 28, 29, 30, 31, 32, 
              // Src: (fdiv:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1) - Complexity = 3
              // Dst: (MUL_IEEE:{ *:[f32] } ?:{ *:[f32] }:$src0, (RECIP_IEEE_eg:{ *:[i32] } ?:{ *:[f32] }:$src1))
/* 36946*/   /*Scope*/ 36|128,1/*164*/, /*->37112*/
/* 36948*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 36950*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36953*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36956*/    OPC_EmitInteger, MVT::i32, 1, 
/* 36959*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36962*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36965*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36968*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36971*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36974*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36977*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 36989*/    OPC_EmitInteger, MVT::i32, 1, 
/* 36992*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36995*/    OPC_EmitInteger, MVT::i32, 0, 
/* 36998*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37001*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37004*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37007*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37010*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 37022*/    OPC_EmitInteger, MVT::i32, 1, 
/* 37025*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 37028*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37031*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37034*/    OPC_EmitNode1, TARGET_VAL(R600::RECIP_IEEE_cm), 0,
                  MVT::i32, 13/*#Ops*/, 12, 13, 14, 15, 1, 16, 17, 18, 19, 20, 21, 22, 23,  // Results = #24
/* 37053*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37056*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37059*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37062*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 37074*/    OPC_EmitInteger, MVT::i32, 1, 
/* 37077*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 37080*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37083*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37086*/    OPC_MorphNodeTo1, TARGET_VAL(R600::MUL_IEEE), 0,
                  MVT::f32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 24, 25, 26, 27, 28, 29, 30, 31, 32, 
              // Src: (fdiv:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1) - Complexity = 3
              // Dst: (MUL_IEEE:{ *:[f32] } ?:{ *:[f32] }:$src0, (RECIP_IEEE_cm:{ *:[i32] } ?:{ *:[f32] }:$src1))
/* 37112*/   0, /*End of Scope*/
/* 37113*/  0, /*End of Scope*/
/* 37114*/ /*SwitchOpcode*/ 39|128,3/*423*/, TARGET_VAL(AMDGPUISD::RCP),// ->37541
/* 37118*/  OPC_Scope, 83|128,1/*211*/, /*->37332*/ // 2 children in Scope
/* 37121*/   OPC_MoveChild0,
/* 37122*/   OPC_CheckOpcode, TARGET_VAL(ISD::FSQRT),
/* 37125*/   OPC_RecordChild0, // #0 = $src
/* 37126*/   OPC_MoveParent,
/* 37127*/   OPC_CheckType, MVT::f32,
/* 37129*/   OPC_Scope, 66, /*->37197*/ // 3 children in Scope
/* 37131*/    OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 37133*/    OPC_EmitInteger, MVT::i32, 1, 
/* 37136*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37139*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37142*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37145*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37148*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37151*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37154*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 37166*/    OPC_EmitInteger, MVT::i32, 1, 
/* 37169*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 37172*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37175*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37178*/    OPC_MorphNodeTo1, TARGET_VAL(R600::RECIPSQRT_IEEE_r600), 0,
                  MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
              // Src: (AMDGPUrcp_impl:{ *:[f32] } (fsqrt:{ *:[f32] } f32:{ *:[f32] }:$src)) - Complexity = 6
              // Dst: (RECIPSQRT_IEEE_r600:{ *:[f32] } ?:{ *:[f32] }:$src)
/* 37197*/   /*Scope*/ 66, /*->37264*/
/* 37198*/    OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 37200*/    OPC_EmitInteger, MVT::i32, 1, 
/* 37203*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37206*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37209*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37212*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37215*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37218*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37221*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 37233*/    OPC_EmitInteger, MVT::i32, 1, 
/* 37236*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 37239*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37242*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37245*/    OPC_MorphNodeTo1, TARGET_VAL(R600::RECIPSQRT_IEEE_eg), 0,
                  MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
              // Src: (AMDGPUrcp_impl:{ *:[f32] } (fsqrt:{ *:[f32] } f32:{ *:[f32] }:$src)) - Complexity = 6
              // Dst: (RECIPSQRT_IEEE_eg:{ *:[f32] } ?:{ *:[f32] }:$src)
/* 37264*/   /*Scope*/ 66, /*->37331*/
/* 37265*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 37267*/    OPC_EmitInteger, MVT::i32, 1, 
/* 37270*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37273*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37276*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37279*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37282*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37285*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37288*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 37300*/    OPC_EmitInteger, MVT::i32, 1, 
/* 37303*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 37306*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37309*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37312*/    OPC_MorphNodeTo1, TARGET_VAL(R600::RECIPSQRT_IEEE_cm), 0,
                  MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
              // Src: (AMDGPUrcp_impl:{ *:[f32] } (fsqrt:{ *:[f32] } f32:{ *:[f32] }:$src)) - Complexity = 6
              // Dst: (RECIPSQRT_IEEE_cm:{ *:[f32] } ?:{ *:[f32] }:$src)
/* 37331*/   0, /*End of Scope*/
/* 37332*/  /*Scope*/ 78|128,1/*206*/, /*->37540*/
/* 37334*/   OPC_RecordChild0, // #0 = $src0
/* 37335*/   OPC_CheckType, MVT::f32,
/* 37337*/   OPC_Scope, 66, /*->37405*/ // 3 children in Scope
/* 37339*/    OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 37341*/    OPC_EmitInteger, MVT::i32, 1, 
/* 37344*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37347*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37350*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37353*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37356*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37359*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37362*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 37374*/    OPC_EmitInteger, MVT::i32, 1, 
/* 37377*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 37380*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37383*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37386*/    OPC_MorphNodeTo1, TARGET_VAL(R600::RECIP_IEEE_r600), 0,
                  MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
              // Src: (AMDGPUrcp_impl:{ *:[f32] } f32:{ *:[f32] }:$src0) - Complexity = 3
              // Dst: (RECIP_IEEE_r600:{ *:[f32] } f32:{ *:[f32] }:$src0)
/* 37405*/   /*Scope*/ 66, /*->37472*/
/* 37406*/    OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 37408*/    OPC_EmitInteger, MVT::i32, 1, 
/* 37411*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37414*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37417*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37420*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37423*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37426*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37429*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 37441*/    OPC_EmitInteger, MVT::i32, 1, 
/* 37444*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 37447*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37450*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37453*/    OPC_MorphNodeTo1, TARGET_VAL(R600::RECIP_IEEE_eg), 0,
                  MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
              // Src: (AMDGPUrcp_impl:{ *:[f32] } f32:{ *:[f32] }:$src0) - Complexity = 3
              // Dst: (RECIP_IEEE_eg:{ *:[f32] } f32:{ *:[f32] }:$src0)
/* 37472*/   /*Scope*/ 66, /*->37539*/
/* 37473*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 37475*/    OPC_EmitInteger, MVT::i32, 1, 
/* 37478*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37481*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37484*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37487*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37490*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37493*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37496*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 37508*/    OPC_EmitInteger, MVT::i32, 1, 
/* 37511*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 37514*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37517*/    OPC_EmitInteger, MVT::i32, 0, 
/* 37520*/    OPC_MorphNodeTo1, TARGET_VAL(R600::RECIP_IEEE_cm), 0,
                  MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
              // Src: (AMDGPUrcp_impl:{ *:[f32] } f32:{ *:[f32] }:$src0) - Complexity = 3
              // Dst: (RECIP_IEEE_cm:{ *:[f32] } f32:{ *:[f32] }:$src0)
/* 37539*/   0, /*End of Scope*/
/* 37540*/  0, /*End of Scope*/
/* 37541*/ /*SwitchOpcode*/ 104, TARGET_VAL(ISD::FADD),// ->37648
/* 37544*/  OPC_RecordChild0, // #0 = $src0
/* 37545*/  OPC_RecordChild1, // #1 = $src1
/* 37546*/  OPC_CheckType, MVT::f32,
/* 37548*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 37550*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37553*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37556*/  OPC_EmitInteger, MVT::i32, 1, 
/* 37559*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37562*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37565*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37568*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37571*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37574*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37577*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 37589*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37592*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37595*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37598*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 37610*/  OPC_EmitInteger, MVT::i32, 1, 
/* 37613*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 37616*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37619*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37622*/  OPC_MorphNodeTo1, TARGET_VAL(R600::ADD), 0,
                MVT::f32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
            // Src: (fadd:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0, R600_Reg32:{ *:[f32] }:$src1) - Complexity = 3
            // Dst: (ADD:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0, R600_Reg32:{ *:[f32] }:$src1)
/* 37648*/ /*SwitchOpcode*/ 104, TARGET_VAL(ISD::FMUL),// ->37755
/* 37651*/  OPC_RecordChild0, // #0 = $src0
/* 37652*/  OPC_RecordChild1, // #1 = $src1
/* 37653*/  OPC_CheckType, MVT::f32,
/* 37655*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 37657*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37660*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37663*/  OPC_EmitInteger, MVT::i32, 1, 
/* 37666*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37669*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37672*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37675*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37678*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37681*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37684*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 37696*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37699*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37702*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37705*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 37717*/  OPC_EmitInteger, MVT::i32, 1, 
/* 37720*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 37723*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37726*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37729*/  OPC_MorphNodeTo1, TARGET_VAL(R600::MUL_IEEE), 0,
                MVT::f32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
            // Src: (fmul:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0, R600_Reg32:{ *:[f32] }:$src1) - Complexity = 3
            // Dst: (MUL_IEEE:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0, R600_Reg32:{ *:[f32] }:$src1)
/* 37755*/ /*SwitchOpcode*/ 104, TARGET_VAL(AMDGPUISD::FMAX_LEGACY),// ->37862
/* 37758*/  OPC_RecordChild0, // #0 = $src0
/* 37759*/  OPC_RecordChild1, // #1 = $src1
/* 37760*/  OPC_CheckType, MVT::f32,
/* 37762*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 37764*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37767*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37770*/  OPC_EmitInteger, MVT::i32, 1, 
/* 37773*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37776*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37779*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37782*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37785*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37788*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37791*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 37803*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37806*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37809*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37812*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 37824*/  OPC_EmitInteger, MVT::i32, 1, 
/* 37827*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 37830*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37833*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37836*/  OPC_MorphNodeTo1, TARGET_VAL(R600::MAX), 0,
                MVT::f32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
            // Src: (AMDGPUfmax_legacy:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0, R600_Reg32:{ *:[f32] }:$src1) - Complexity = 3
            // Dst: (MAX:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0, R600_Reg32:{ *:[f32] }:$src1)
/* 37862*/ /*SwitchOpcode*/ 104, TARGET_VAL(AMDGPUISD::FMIN_LEGACY),// ->37969
/* 37865*/  OPC_RecordChild0, // #0 = $src0
/* 37866*/  OPC_RecordChild1, // #1 = $src1
/* 37867*/  OPC_CheckType, MVT::f32,
/* 37869*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 37871*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37874*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37877*/  OPC_EmitInteger, MVT::i32, 1, 
/* 37880*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37883*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37886*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37889*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37892*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37895*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37898*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 37910*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37913*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37916*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37919*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 37931*/  OPC_EmitInteger, MVT::i32, 1, 
/* 37934*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 37937*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37940*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37943*/  OPC_MorphNodeTo1, TARGET_VAL(R600::MIN), 0,
                MVT::f32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
            // Src: (AMDGPUfmin_legacy:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0, R600_Reg32:{ *:[f32] }:$src1) - Complexity = 3
            // Dst: (MIN:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0, R600_Reg32:{ *:[f32] }:$src1)
/* 37969*/ /*SwitchOpcode*/ 104, TARGET_VAL(ISD::FMAXNUM),// ->38076
/* 37972*/  OPC_RecordChild0, // #0 = $src0
/* 37973*/  OPC_RecordChild1, // #1 = $src1
/* 37974*/  OPC_CheckType, MVT::f32,
/* 37976*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 37978*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37981*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37984*/  OPC_EmitInteger, MVT::i32, 1, 
/* 37987*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37990*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37993*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37996*/  OPC_EmitInteger, MVT::i32, 0, 
/* 37999*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38002*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38005*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 38017*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38020*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38023*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38026*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 38038*/  OPC_EmitInteger, MVT::i32, 1, 
/* 38041*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 38044*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38047*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38050*/  OPC_MorphNodeTo1, TARGET_VAL(R600::MAX_DX10), 0,
                MVT::f32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
            // Src: (fmaxnum:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0, R600_Reg32:{ *:[f32] }:$src1) - Complexity = 3
            // Dst: (MAX_DX10:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0, R600_Reg32:{ *:[f32] }:$src1)
/* 38076*/ /*SwitchOpcode*/ 104, TARGET_VAL(ISD::FMINNUM),// ->38183
/* 38079*/  OPC_RecordChild0, // #0 = $src0
/* 38080*/  OPC_RecordChild1, // #1 = $src1
/* 38081*/  OPC_CheckType, MVT::f32,
/* 38083*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 38085*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38088*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38091*/  OPC_EmitInteger, MVT::i32, 1, 
/* 38094*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38097*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38100*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38103*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38106*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38109*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38112*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 38124*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38127*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38130*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38133*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 38145*/  OPC_EmitInteger, MVT::i32, 1, 
/* 38148*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 38151*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38154*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38157*/  OPC_MorphNodeTo1, TARGET_VAL(R600::MIN_DX10), 0,
                MVT::f32, 20/*#Ops*/, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 19, 
            // Src: (fminnum:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0, R600_Reg32:{ *:[f32] }:$src1) - Complexity = 3
            // Dst: (MIN_DX10:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0, R600_Reg32:{ *:[f32] }:$src1)
/* 38183*/ /*SwitchOpcode*/ 69, TARGET_VAL(AMDGPUISD::FRACT),// ->38255
/* 38186*/  OPC_RecordChild0, // #0 = $src0
/* 38187*/  OPC_CheckType, MVT::f32,
/* 38189*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 38191*/  OPC_EmitInteger, MVT::i32, 1, 
/* 38194*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38197*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38200*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38203*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38206*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38209*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38212*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 38224*/  OPC_EmitInteger, MVT::i32, 1, 
/* 38227*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 38230*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38233*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38236*/  OPC_MorphNodeTo1, TARGET_VAL(R600::FRACT), 0,
                MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
            // Src: (AMDGPUfract_impl:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
            // Dst: (FRACT:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 38255*/ /*SwitchOpcode*/ 69, TARGET_VAL(ISD::FTRUNC),// ->38327
/* 38258*/  OPC_RecordChild0, // #0 = $src0
/* 38259*/  OPC_CheckType, MVT::f32,
/* 38261*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 38263*/  OPC_EmitInteger, MVT::i32, 1, 
/* 38266*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38269*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38272*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38275*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38278*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38281*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38284*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 38296*/  OPC_EmitInteger, MVT::i32, 1, 
/* 38299*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 38302*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38305*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38308*/  OPC_MorphNodeTo1, TARGET_VAL(R600::TRUNC), 0,
                MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
            // Src: (ftrunc:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
            // Dst: (TRUNC:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 38327*/ /*SwitchOpcode*/ 69, TARGET_VAL(ISD::FCEIL),// ->38399
/* 38330*/  OPC_RecordChild0, // #0 = $src0
/* 38331*/  OPC_CheckType, MVT::f32,
/* 38333*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 38335*/  OPC_EmitInteger, MVT::i32, 1, 
/* 38338*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38341*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38344*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38347*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38350*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38353*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38356*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 38368*/  OPC_EmitInteger, MVT::i32, 1, 
/* 38371*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 38374*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38377*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38380*/  OPC_MorphNodeTo1, TARGET_VAL(R600::CEIL), 0,
                MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
            // Src: (fceil:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
            // Dst: (CEIL:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 38399*/ /*SwitchOpcode*/ 69, TARGET_VAL(ISD::FRINT),// ->38471
/* 38402*/  OPC_RecordChild0, // #0 = $src0
/* 38403*/  OPC_CheckType, MVT::f32,
/* 38405*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 38407*/  OPC_EmitInteger, MVT::i32, 1, 
/* 38410*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38413*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38416*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38419*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38422*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38425*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38428*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 38440*/  OPC_EmitInteger, MVT::i32, 1, 
/* 38443*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 38446*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38449*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38452*/  OPC_MorphNodeTo1, TARGET_VAL(R600::RNDNE), 0,
                MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
            // Src: (frint:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
            // Dst: (RNDNE:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 38471*/ /*SwitchOpcode*/ 69, TARGET_VAL(ISD::FFLOOR),// ->38543
/* 38474*/  OPC_RecordChild0, // #0 = $src0
/* 38475*/  OPC_CheckType, MVT::f32,
/* 38477*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 38479*/  OPC_EmitInteger, MVT::i32, 1, 
/* 38482*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38485*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38488*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38491*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38494*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38497*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38500*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 38512*/  OPC_EmitInteger, MVT::i32, 1, 
/* 38515*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 38518*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38521*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38524*/  OPC_MorphNodeTo1, TARGET_VAL(R600::FLOOR), 0,
                MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
            // Src: (ffloor:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
            // Dst: (FLOOR:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 38543*/ /*SwitchOpcode*/ 90|128,2/*346*/, TARGET_VAL(AMDGPUISD::DOT4),// ->38893
/* 38547*/  OPC_RecordChild0, // #0 = $src0_X
/* 38548*/  OPC_RecordChild1, // #1 = $src1_X
/* 38549*/  OPC_RecordChild2, // #2 = $src0_Y
/* 38550*/  OPC_RecordChild3, // #3 = $src1_Y
/* 38551*/  OPC_RecordChild4, // #4 = $src0_Z
/* 38552*/  OPC_RecordChild5, // #5 = $src1_Z
/* 38553*/  OPC_RecordChild6, // #6 = $src0_W
/* 38554*/  OPC_RecordChild7, // #7 = $src1_W
/* 38555*/  OPC_CheckType, MVT::f32,
/* 38557*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 38559*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38562*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38565*/  OPC_EmitInteger, MVT::i32, 1, 
/* 38568*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38571*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38574*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38577*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38580*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38583*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38586*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 38598*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38601*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38604*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38607*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 38619*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 38622*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38625*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38628*/  OPC_EmitInteger, MVT::i32, 1, 
/* 38631*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38634*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38637*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38640*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38643*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38646*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38649*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 38661*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38664*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38667*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38670*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 38682*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 38685*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38688*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38691*/  OPC_EmitInteger, MVT::i32, 1, 
/* 38694*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38697*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38700*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38703*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38706*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38709*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38712*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 38724*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38727*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38730*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38733*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 38745*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 38748*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38751*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38754*/  OPC_EmitInteger, MVT::i32, 1, 
/* 38757*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38760*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38763*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38766*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38769*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38772*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38775*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 38787*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38790*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38793*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38796*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 38808*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 38811*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38814*/  OPC_EmitInteger, MVT::i32, 0, 
/* 38817*/  OPC_MorphNodeTo1, TARGET_VAL(R600::DOT_4), 0,
                MVT::f32, 70/*#Ops*/, 8, 9, 10, 11, 12, 13, 0, 14, 15, 16, 17, 1, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 2, 29, 30, 31, 32, 3, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 4, 44, 45, 46, 47, 5, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 6, 59, 60, 61, 62, 7, 63, 64, 65, 66, 67, 68, 69, 
            // Src: (DOT4:{ *:[f32] } R600_TReg32_X:{ *:[f32] }:$src0_X, R600_TReg32_X:{ *:[f32] }:$src1_X, R600_TReg32_Y:{ *:[f32] }:$src0_Y, R600_TReg32_Y:{ *:[f32] }:$src1_Y, R600_TReg32_Z:{ *:[f32] }:$src0_Z, R600_TReg32_Z:{ *:[f32] }:$src1_Z, R600_TReg32_W:{ *:[f32] }:$src0_W, R600_TReg32_W:{ *:[f32] }:$src1_W) - Complexity = 3
            // Dst: (DOT_4:{ *:[f32] } R600_TReg32_X:{ *:[f32] }:$src0_X, R600_TReg32_X:{ *:[f32] }:$src1_X, R600_TReg32_Y:{ *:[f32] }:$src0_Y, R600_TReg32_Y:{ *:[f32] }:$src1_Y, R600_TReg32_Z:{ *:[f32] }:$src0_Z, R600_TReg32_Z:{ *:[f32] }:$src1_Z, R600_TReg32_W:{ *:[f32] }:$src0_W, R600_TReg32_W:{ *:[f32] }:$src1_W)
/* 38893*/ /*SwitchOpcode*/ 77|128,1/*205*/, TARGET_VAL(ISD::FMAD),// ->39102
/* 38897*/  OPC_RecordChild0, // #0 = $src0
/* 38898*/  OPC_RecordChild1, // #1 = $src1
/* 38899*/  OPC_RecordChild2, // #2 = $src2
/* 38900*/  OPC_CheckType, MVT::f32,
/* 38902*/  OPC_Scope, 98, /*->39002*/ // 2 children in Scope
/* 38904*/   OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 38906*/   OPC_EmitInteger, MVT::i32, 0, 
/* 38909*/   OPC_EmitInteger, MVT::i32, 0, 
/* 38912*/   OPC_EmitInteger, MVT::i32, 0, 
/* 38915*/   OPC_EmitInteger, MVT::i32, 0, 
/* 38918*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 38930*/   OPC_EmitInteger, MVT::i32, 0, 
/* 38933*/   OPC_EmitInteger, MVT::i32, 0, 
/* 38936*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 38948*/   OPC_EmitInteger, MVT::i32, 0, 
/* 38951*/   OPC_EmitInteger, MVT::i32, 0, 
/* 38954*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 38966*/   OPC_EmitInteger, MVT::i32, 1, 
/* 38969*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 38972*/   OPC_EmitInteger, MVT::i32, 0, 
/* 38975*/   OPC_EmitInteger, MVT::i32, 0, 
/* 38978*/   OPC_MorphNodeTo1, TARGET_VAL(R600::MULADD_IEEE_r600), 0,
                 MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
             // Src: (fmad:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2) - Complexity = 3
             // Dst: (MULADD_IEEE_r600:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2)
/* 39002*/  /*Scope*/ 98, /*->39101*/
/* 39003*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 39005*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39008*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39011*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39014*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39017*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 39029*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39032*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39035*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 39047*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39050*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39053*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 39065*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39068*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 39071*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39074*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39077*/   OPC_MorphNodeTo1, TARGET_VAL(R600::MULADD_IEEE_eg), 0,
                 MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
             // Src: (fmad:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2) - Complexity = 3
             // Dst: (MULADD_IEEE_eg:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2)
/* 39101*/  0, /*End of Scope*/
/* 39102*/ /*SwitchOpcode*/ 77|128,1/*205*/, TARGET_VAL(AMDGPUISD::FMAD_FTZ),// ->39311
/* 39106*/  OPC_RecordChild0, // #0 = $src0
/* 39107*/  OPC_RecordChild1, // #1 = $src1
/* 39108*/  OPC_RecordChild2, // #2 = $src2
/* 39109*/  OPC_CheckType, MVT::f32,
/* 39111*/  OPC_Scope, 98, /*->39211*/ // 2 children in Scope
/* 39113*/   OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 39115*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39118*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39121*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39124*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39127*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 39139*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39142*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39145*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 39157*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39160*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39163*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 39175*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39178*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 39181*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39184*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39187*/   OPC_MorphNodeTo1, TARGET_VAL(R600::MULADD_IEEE_r600), 0,
                 MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
             // Src: (AMDGPUfmad_ftz_impl:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2) - Complexity = 3
             // Dst: (MULADD_IEEE_r600:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2)
/* 39211*/  /*Scope*/ 98, /*->39310*/
/* 39212*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 39214*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39217*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39220*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39223*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39226*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 39238*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39241*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39244*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 39256*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39259*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39262*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 39274*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39277*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 39280*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39283*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39286*/   OPC_MorphNodeTo1, TARGET_VAL(R600::MULADD_IEEE_eg), 0,
                 MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
             // Src: (AMDGPUfmad_ftz_impl:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2) - Complexity = 3
             // Dst: (MULADD_IEEE_eg:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2)
/* 39310*/  0, /*End of Scope*/
/* 39311*/ /*SwitchOpcode*/ 78|128,1/*206*/, TARGET_VAL(ISD::FEXP2),// ->39521
/* 39315*/  OPC_RecordChild0, // #0 = $src0
/* 39316*/  OPC_CheckType, MVT::f32,
/* 39318*/  OPC_Scope, 66, /*->39386*/ // 3 children in Scope
/* 39320*/   OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 39322*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39325*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39328*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39331*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39334*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39337*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39340*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39343*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 39355*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39358*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 39361*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39364*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39367*/   OPC_MorphNodeTo1, TARGET_VAL(R600::EXP_IEEE_r600), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (fexp2:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (EXP_IEEE_r600:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 39386*/  /*Scope*/ 66, /*->39453*/
/* 39387*/   OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 39389*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39392*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39395*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39398*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39401*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39404*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39407*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39410*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 39422*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39425*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 39428*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39431*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39434*/   OPC_MorphNodeTo1, TARGET_VAL(R600::EXP_IEEE_eg), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (fexp2:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (EXP_IEEE_eg:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 39453*/  /*Scope*/ 66, /*->39520*/
/* 39454*/   OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 39456*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39459*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39462*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39465*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39468*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39471*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39474*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39477*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 39489*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39492*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 39495*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39498*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39501*/   OPC_MorphNodeTo1, TARGET_VAL(R600::EXP_IEEE_cm), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (fexp2:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (EXP_IEEE_cm:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 39520*/  0, /*End of Scope*/
/* 39521*/ /*SwitchOpcode*/ 78|128,1/*206*/, TARGET_VAL(ISD::FLOG2),// ->39731
/* 39525*/  OPC_RecordChild0, // #0 = $src0
/* 39526*/  OPC_CheckType, MVT::f32,
/* 39528*/  OPC_Scope, 66, /*->39596*/ // 3 children in Scope
/* 39530*/   OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 39532*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39535*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39538*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39541*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39544*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39547*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39550*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39553*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 39565*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39568*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 39571*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39574*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39577*/   OPC_MorphNodeTo1, TARGET_VAL(R600::LOG_IEEE_r600), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (flog2:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (LOG_IEEE_r600:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 39596*/  /*Scope*/ 66, /*->39663*/
/* 39597*/   OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 39599*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39602*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39605*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39608*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39611*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39614*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39617*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39620*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 39632*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39635*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 39638*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39641*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39644*/   OPC_MorphNodeTo1, TARGET_VAL(R600::LOG_IEEE_eg), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (flog2:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (LOG_IEEE_eg:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 39663*/  /*Scope*/ 66, /*->39730*/
/* 39664*/   OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 39666*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39669*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39672*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39675*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39678*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39681*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39684*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39687*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 39699*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39702*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 39705*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39708*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39711*/   OPC_MorphNodeTo1, TARGET_VAL(R600::LOG_IEEE_cm), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (flog2:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (LOG_IEEE_cm:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 39730*/  0, /*End of Scope*/
/* 39731*/ /*SwitchOpcode*/ 78|128,1/*206*/, TARGET_VAL(AMDGPUISD::RSQ_CLAMP),// ->39941
/* 39735*/  OPC_RecordChild0, // #0 = $src0
/* 39736*/  OPC_CheckType, MVT::f32,
/* 39738*/  OPC_Scope, 66, /*->39806*/ // 3 children in Scope
/* 39740*/   OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 39742*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39745*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39748*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39751*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39754*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39757*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39760*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39763*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 39775*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39778*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 39781*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39784*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39787*/   OPC_MorphNodeTo1, TARGET_VAL(R600::RECIPSQRT_CLAMPED_r600), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (AMDGPUrsq_clamp_impl:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (RECIPSQRT_CLAMPED_r600:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 39806*/  /*Scope*/ 66, /*->39873*/
/* 39807*/   OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 39809*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39812*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39815*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39818*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39821*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39824*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39827*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39830*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 39842*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39845*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 39848*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39851*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39854*/   OPC_MorphNodeTo1, TARGET_VAL(R600::RECIPSQRT_CLAMPED_eg), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (AMDGPUrsq_clamp_impl:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (RECIPSQRT_CLAMPED_eg:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 39873*/  /*Scope*/ 66, /*->39940*/
/* 39874*/   OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 39876*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39879*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39882*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39885*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39888*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39891*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39894*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39897*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 39909*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39912*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 39915*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39918*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39921*/   OPC_MorphNodeTo1, TARGET_VAL(R600::RECIPSQRT_CLAMPED_cm), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (AMDGPUrsq_clamp_impl:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (RECIPSQRT_CLAMPED_cm:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 39940*/  0, /*End of Scope*/
/* 39941*/ /*SwitchOpcode*/ 78|128,1/*206*/, TARGET_VAL(AMDGPUISD::RSQ),// ->40151
/* 39945*/  OPC_RecordChild0, // #0 = $src0
/* 39946*/  OPC_CheckType, MVT::f32,
/* 39948*/  OPC_Scope, 66, /*->40016*/ // 3 children in Scope
/* 39950*/   OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 39952*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39955*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39958*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39961*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39964*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39967*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39970*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39973*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 39985*/   OPC_EmitInteger, MVT::i32, 1, 
/* 39988*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 39991*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39994*/   OPC_EmitInteger, MVT::i32, 0, 
/* 39997*/   OPC_MorphNodeTo1, TARGET_VAL(R600::RECIPSQRT_IEEE_r600), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (AMDGPUrsq_impl:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (RECIPSQRT_IEEE_r600:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 40016*/  /*Scope*/ 66, /*->40083*/
/* 40017*/   OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 40019*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40022*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40025*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40028*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40031*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40034*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40037*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40040*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 40052*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40055*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 40058*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40061*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40064*/   OPC_MorphNodeTo1, TARGET_VAL(R600::RECIPSQRT_IEEE_eg), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (AMDGPUrsq_impl:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (RECIPSQRT_IEEE_eg:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 40083*/  /*Scope*/ 66, /*->40150*/
/* 40084*/   OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 40086*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40089*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40092*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40095*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40098*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40101*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40104*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40107*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 40119*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40122*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 40125*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40128*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40131*/   OPC_MorphNodeTo1, TARGET_VAL(R600::RECIPSQRT_IEEE_cm), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (AMDGPUrsq_impl:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (RECIPSQRT_IEEE_cm:{ *:[f32] } R600_Reg32:{ *:[f32] }:$src0)
/* 40150*/  0, /*End of Scope*/
/* 40151*/ /*SwitchOpcode*/ 13|128,1/*141*/, TARGET_VAL(ISD::SINT_TO_FP),// ->40296
/* 40155*/  OPC_RecordChild0, // #0 = $src0
/* 40156*/  OPC_CheckChild0Type, MVT::i32,
/* 40158*/  OPC_CheckType, MVT::f32,
/* 40160*/  OPC_Scope, 66, /*->40228*/ // 2 children in Scope
/* 40162*/   OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 40164*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40167*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40170*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40173*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40176*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40179*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40182*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40185*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 40197*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40200*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 40203*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40206*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40209*/   OPC_MorphNodeTo1, TARGET_VAL(R600::INT_TO_FLT_r600), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (sint_to_fp:{ *:[f32] } R600_Reg32:{ *:[i32] }:$src0) - Complexity = 3
             // Dst: (INT_TO_FLT_r600:{ *:[f32] } R600_Reg32:{ *:[i32] }:$src0)
/* 40228*/  /*Scope*/ 66, /*->40295*/
/* 40229*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 40231*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40234*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40237*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40240*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40243*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40246*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40249*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40252*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 40264*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40267*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 40270*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40273*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40276*/   OPC_MorphNodeTo1, TARGET_VAL(R600::INT_TO_FLT_eg), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (sint_to_fp:{ *:[f32] } R600_Reg32:{ *:[i32] }:$src0) - Complexity = 3
             // Dst: (INT_TO_FLT_eg:{ *:[f32] } R600_Reg32:{ *:[i32] }:$src0)
/* 40295*/  0, /*End of Scope*/
/* 40296*/ /*SwitchOpcode*/ 13|128,1/*141*/, TARGET_VAL(ISD::UINT_TO_FP),// ->40441
/* 40300*/  OPC_RecordChild0, // #0 = $src0
/* 40301*/  OPC_CheckChild0Type, MVT::i32,
/* 40303*/  OPC_CheckType, MVT::f32,
/* 40305*/  OPC_Scope, 66, /*->40373*/ // 2 children in Scope
/* 40307*/   OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 40309*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40312*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40315*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40318*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40321*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40324*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40327*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40330*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 40342*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40345*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 40348*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40351*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40354*/   OPC_MorphNodeTo1, TARGET_VAL(R600::UINT_TO_FLT_r600), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (uint_to_fp:{ *:[f32] } R600_Reg32:{ *:[i32] }:$src0) - Complexity = 3
             // Dst: (UINT_TO_FLT_r600:{ *:[f32] } R600_Reg32:{ *:[i32] }:$src0)
/* 40373*/  /*Scope*/ 66, /*->40440*/
/* 40374*/   OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 40376*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40379*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40382*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40385*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40388*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40391*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40394*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40397*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 40409*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40412*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 40415*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40418*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40421*/   OPC_MorphNodeTo1, TARGET_VAL(R600::UINT_TO_FLT_eg), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (uint_to_fp:{ *:[f32] } R600_Reg32:{ *:[i32] }:$src0) - Complexity = 3
             // Dst: (UINT_TO_FLT_eg:{ *:[f32] } R600_Reg32:{ *:[i32] }:$src0)
/* 40440*/  0, /*End of Scope*/
/* 40441*/ /*SwitchOpcode*/ 19|128,2/*275*/, TARGET_VAL(AMDGPUISD::SIN_HW),// ->40720
/* 40445*/  OPC_RecordChild0, // #0 = $src0
/* 40446*/  OPC_CheckChild0Type, MVT::f32,
/* 40448*/  OPC_CheckType, MVT::f32,
/* 40450*/  OPC_Scope, 66, /*->40518*/ // 4 children in Scope
/* 40452*/   OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 40454*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40457*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40460*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40463*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40466*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40469*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40472*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40475*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 40487*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40490*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 40493*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40496*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40499*/   OPC_MorphNodeTo1, TARGET_VAL(R600::SIN_r600), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (SIN_HW:{ *:[f32] } f32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (SIN_r600:{ *:[f32] } f32:{ *:[f32] }:$src0)
/* 40518*/  /*Scope*/ 66, /*->40585*/
/* 40519*/   OPC_CheckPatternPredicate, 5, // (Subtarget->getGeneration() == AMDGPUSubtarget::R700)
/* 40521*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40524*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40527*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40530*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40533*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40536*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40539*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40542*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 40554*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40557*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 40560*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40563*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40566*/   OPC_MorphNodeTo1, TARGET_VAL(R600::SIN_r700), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (SIN_HW:{ *:[f32] } f32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (SIN_r700:{ *:[f32] } f32:{ *:[f32] }:$src0)
/* 40585*/  /*Scope*/ 66, /*->40652*/
/* 40586*/   OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 40588*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40591*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40594*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40597*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40600*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40603*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40606*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40609*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 40621*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40624*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 40627*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40630*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40633*/   OPC_MorphNodeTo1, TARGET_VAL(R600::SIN_eg), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (SIN_HW:{ *:[f32] } f32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (SIN_eg:{ *:[f32] } f32:{ *:[f32] }:$src0)
/* 40652*/  /*Scope*/ 66, /*->40719*/
/* 40653*/   OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 40655*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40658*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40661*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40664*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40667*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40670*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40673*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40676*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 40688*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40691*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 40694*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40697*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40700*/   OPC_MorphNodeTo1, TARGET_VAL(R600::SIN_cm), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (SIN_HW:{ *:[f32] } f32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (SIN_cm:{ *:[f32] } f32:{ *:[f32] }:$src0)
/* 40719*/  0, /*End of Scope*/
/* 40720*/ /*SwitchOpcode*/ 19|128,2/*275*/, TARGET_VAL(AMDGPUISD::COS_HW),// ->40999
/* 40724*/  OPC_RecordChild0, // #0 = $src0
/* 40725*/  OPC_CheckChild0Type, MVT::f32,
/* 40727*/  OPC_CheckType, MVT::f32,
/* 40729*/  OPC_Scope, 66, /*->40797*/ // 4 children in Scope
/* 40731*/   OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 40733*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40736*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40739*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40742*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40745*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40748*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40751*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40754*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 40766*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40769*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 40772*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40775*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40778*/   OPC_MorphNodeTo1, TARGET_VAL(R600::COS_r600), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (COS_HW:{ *:[f32] } f32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (COS_r600:{ *:[f32] } f32:{ *:[f32] }:$src0)
/* 40797*/  /*Scope*/ 66, /*->40864*/
/* 40798*/   OPC_CheckPatternPredicate, 5, // (Subtarget->getGeneration() == AMDGPUSubtarget::R700)
/* 40800*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40803*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40806*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40809*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40812*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40815*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40818*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40821*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 40833*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40836*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 40839*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40842*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40845*/   OPC_MorphNodeTo1, TARGET_VAL(R600::COS_r700), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (COS_HW:{ *:[f32] } f32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (COS_r700:{ *:[f32] } f32:{ *:[f32] }:$src0)
/* 40864*/  /*Scope*/ 66, /*->40931*/
/* 40865*/   OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 40867*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40870*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40873*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40876*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40879*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40882*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40885*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40888*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 40900*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40903*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 40906*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40909*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40912*/   OPC_MorphNodeTo1, TARGET_VAL(R600::COS_eg), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (COS_HW:{ *:[f32] } f32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (COS_eg:{ *:[f32] } f32:{ *:[f32] }:$src0)
/* 40931*/  /*Scope*/ 66, /*->40998*/
/* 40932*/   OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 40934*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40937*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40940*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40943*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40946*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40949*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40952*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40955*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 40967*/   OPC_EmitInteger, MVT::i32, 1, 
/* 40970*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 40973*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40976*/   OPC_EmitInteger, MVT::i32, 0, 
/* 40979*/   OPC_MorphNodeTo1, TARGET_VAL(R600::COS_cm), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
             // Src: (COS_HW:{ *:[f32] } f32:{ *:[f32] }:$src0) - Complexity = 3
             // Dst: (COS_cm:{ *:[f32] } f32:{ *:[f32] }:$src0)
/* 40998*/  0, /*End of Scope*/
/* 40999*/ /*SwitchOpcode*/ 103, TARGET_VAL(ISD::FMA),// ->41105
/* 41002*/  OPC_RecordChild0, // #0 = $src0
/* 41003*/  OPC_RecordChild1, // #1 = $src1
/* 41004*/  OPC_RecordChild2, // #2 = $src2
/* 41005*/  OPC_CheckType, MVT::f32,
/* 41007*/  OPC_CheckPatternPredicate, 6, // (Subtarget->hasFMA()) && (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 41009*/  OPC_EmitInteger, MVT::i32, 0, 
/* 41012*/  OPC_EmitInteger, MVT::i32, 0, 
/* 41015*/  OPC_EmitInteger, MVT::i32, 0, 
/* 41018*/  OPC_EmitInteger, MVT::i32, 0, 
/* 41021*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 41033*/  OPC_EmitInteger, MVT::i32, 0, 
/* 41036*/  OPC_EmitInteger, MVT::i32, 0, 
/* 41039*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 41051*/  OPC_EmitInteger, MVT::i32, 0, 
/* 41054*/  OPC_EmitInteger, MVT::i32, 0, 
/* 41057*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 41069*/  OPC_EmitInteger, MVT::i32, 1, 
/* 41072*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 41075*/  OPC_EmitInteger, MVT::i32, 0, 
/* 41078*/  OPC_EmitInteger, MVT::i32, 0, 
/* 41081*/  OPC_MorphNodeTo1, TARGET_VAL(R600::FMA_eg), 0,
                MVT::f32, 18/*#Ops*/, 3, 4, 0, 5, 6, 7, 1, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 
            // Src: (fma:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2) - Complexity = 3
            // Dst: (FMA_eg:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1, f32:{ *:[f32] }:$src2)
/* 41105*/ /*SwitchOpcode*/ 71, TARGET_VAL(ISD::FP16_TO_FP),// ->41179
/* 41108*/  OPC_RecordChild0, // #0 = $src0
/* 41109*/  OPC_CheckChild0Type, MVT::i32,
/* 41111*/  OPC_CheckType, MVT::f32,
/* 41113*/  OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 41115*/  OPC_EmitInteger, MVT::i32, 1, 
/* 41118*/  OPC_EmitInteger, MVT::i32, 0, 
/* 41121*/  OPC_EmitInteger, MVT::i32, 0, 
/* 41124*/  OPC_EmitInteger, MVT::i32, 0, 
/* 41127*/  OPC_EmitInteger, MVT::i32, 0, 
/* 41130*/  OPC_EmitInteger, MVT::i32, 0, 
/* 41133*/  OPC_EmitInteger, MVT::i32, 0, 
/* 41136*/  OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 41148*/  OPC_EmitInteger, MVT::i32, 1, 
/* 41151*/  OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 41154*/  OPC_EmitInteger, MVT::i32, 0, 
/* 41157*/  OPC_EmitInteger, MVT::i32, 0, 
/* 41160*/  OPC_MorphNodeTo1, TARGET_VAL(R600::FLT16_TO_FLT32), 0,
                MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 
            // Src: (f16_to_fp:{ *:[f32] } R600_Reg32:{ *:[i32] }:$src0) - Complexity = 3
            // Dst: (FLT16_TO_FLT32:{ *:[f32] } R600_Reg32:{ *:[i32] }:$src0)
/* 41179*/ /*SwitchOpcode*/ 17|128,3/*401*/, TARGET_VAL(ISD::FSQRT),// ->41584
/* 41183*/  OPC_RecordChild0, // #0 = $src
/* 41184*/  OPC_CheckType, MVT::f32,
/* 41186*/  OPC_Scope, 2|128,1/*130*/, /*->41319*/ // 3 children in Scope
/* 41189*/   OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 41191*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41194*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41197*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41200*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41203*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41206*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41209*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41212*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41215*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41218*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41221*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41224*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 41236*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41239*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 41242*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41245*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41248*/   OPC_EmitNode1, TARGET_VAL(R600::RECIPSQRT_IEEE_r600), 0,
                 MVT::i32, 13/*#Ops*/, 5, 6, 7, 8, 0, 9, 10, 11, 12, 13, 14, 15, 16,  // Results = #17
/* 41267*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41270*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41273*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41276*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 41288*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41291*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 41294*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41297*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41300*/   OPC_MorphNodeTo1, TARGET_VAL(R600::RECIP_IEEE_r600), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
             // Src: (fsqrt:{ *:[f32] } f32:{ *:[f32] }:$src) - Complexity = 3
             // Dst: (RECIP_IEEE_r600:{ *:[f32] } (RECIPSQRT_IEEE_r600:{ *:[i32] } ?:{ *:[f32] }:$src))
/* 41319*/  /*Scope*/ 2|128,1/*130*/, /*->41451*/
/* 41321*/   OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 41323*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41326*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41329*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41332*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41335*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41338*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41341*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41344*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41347*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41350*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41353*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41356*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 41368*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41371*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 41374*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41377*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41380*/   OPC_EmitNode1, TARGET_VAL(R600::RECIPSQRT_IEEE_eg), 0,
                 MVT::i32, 13/*#Ops*/, 5, 6, 7, 8, 0, 9, 10, 11, 12, 13, 14, 15, 16,  // Results = #17
/* 41399*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41402*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41405*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41408*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 41420*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41423*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 41426*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41429*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41432*/   OPC_MorphNodeTo1, TARGET_VAL(R600::RECIP_IEEE_eg), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
             // Src: (fsqrt:{ *:[f32] } f32:{ *:[f32] }:$src) - Complexity = 3
             // Dst: (RECIP_IEEE_eg:{ *:[f32] } (RECIPSQRT_IEEE_eg:{ *:[i32] } ?:{ *:[f32] }:$src))
/* 41451*/  /*Scope*/ 2|128,1/*130*/, /*->41583*/
/* 41453*/   OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 41455*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41458*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41461*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41464*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41467*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41470*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41473*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41476*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41479*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41482*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41485*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41488*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 41500*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41503*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 41506*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41509*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41512*/   OPC_EmitNode1, TARGET_VAL(R600::RECIPSQRT_IEEE_cm), 0,
                 MVT::i32, 13/*#Ops*/, 5, 6, 7, 8, 0, 9, 10, 11, 12, 13, 14, 15, 16,  // Results = #17
/* 41531*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41534*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41537*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41540*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 41552*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41555*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 41558*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41561*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41564*/   OPC_MorphNodeTo1, TARGET_VAL(R600::RECIP_IEEE_cm), 0,
                 MVT::f32, 13/*#Ops*/, 1, 2, 3, 4, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
             // Src: (fsqrt:{ *:[f32] } f32:{ *:[f32] }:$src) - Complexity = 3
             // Dst: (RECIP_IEEE_cm:{ *:[f32] } (RECIPSQRT_IEEE_cm:{ *:[i32] } ?:{ *:[f32] }:$src))
/* 41583*/  0, /*End of Scope*/
/* 41584*/ /*SwitchOpcode*/ 56|128,5/*696*/, TARGET_VAL(ISD::FPOW),// ->42284
/* 41588*/  OPC_RecordChild0, // #0 = $src0
/* 41589*/  OPC_RecordChild1, // #1 = $src1
/* 41590*/  OPC_CheckType, MVT::f32,
/* 41592*/  OPC_Scope, 100|128,1/*228*/, /*->41823*/ // 3 children in Scope
/* 41595*/   OPC_CheckPatternPredicate, 1, // (Subtarget->getGeneration() <= AMDGPUSubtarget::R700)
/* 41597*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41600*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41603*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41606*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41609*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41612*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41615*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41618*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41621*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41624*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41627*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41630*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41633*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41636*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 41648*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41651*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41654*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41657*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41660*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41663*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41666*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41669*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 41681*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41684*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 41687*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41690*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41693*/   OPC_EmitNode1, TARGET_VAL(R600::LOG_IEEE_r600), 0,
                 MVT::i32, 13/*#Ops*/, 16, 17, 18, 19, 0, 20, 21, 22, 23, 24, 25, 26, 27,  // Results = #28
/* 41712*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41715*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41718*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41721*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 41733*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41736*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 41739*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41742*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41745*/   OPC_EmitNode1, TARGET_VAL(R600::MUL), 0,
                 MVT::i32, 20/*#Ops*/, 6, 7, 8, 9, 10, 11, 1, 12, 13, 14, 15, 28, 29, 30, 31, 32, 33, 34, 35, 36,  // Results = #37
/* 41771*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41774*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41777*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41780*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 41792*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41795*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 41798*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41801*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41804*/   OPC_MorphNodeTo1, TARGET_VAL(R600::EXP_IEEE_r600), 0,
                 MVT::f32, 13/*#Ops*/, 2, 3, 4, 5, 37, 38, 39, 40, 41, 42, 43, 44, 45, 
             // Src: (fpow:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1) - Complexity = 3
             // Dst: (EXP_IEEE_r600:{ *:[f32] } (MUL:{ *:[i32] } f32:{ *:[f32] }:$src1, (LOG_IEEE_r600:{ *:[i32] } f32:{ *:[f32] }:$src0)))
/* 41823*/  /*Scope*/ 100|128,1/*228*/, /*->42053*/
/* 41825*/   OPC_CheckPatternPredicate, 4, // (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA())
/* 41827*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41830*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41833*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41836*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41839*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41842*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41845*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41848*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41851*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41854*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41857*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41860*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41863*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41866*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 41878*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41881*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41884*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41887*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41890*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41893*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41896*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41899*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 41911*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41914*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 41917*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41920*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41923*/   OPC_EmitNode1, TARGET_VAL(R600::LOG_IEEE_eg), 0,
                 MVT::i32, 13/*#Ops*/, 16, 17, 18, 19, 0, 20, 21, 22, 23, 24, 25, 26, 27,  // Results = #28
/* 41942*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41945*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41948*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41951*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 41963*/   OPC_EmitInteger, MVT::i32, 1, 
/* 41966*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 41969*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41972*/   OPC_EmitInteger, MVT::i32, 0, 
/* 41975*/   OPC_EmitNode1, TARGET_VAL(R600::MUL), 0,
                 MVT::i32, 20/*#Ops*/, 6, 7, 8, 9, 10, 11, 1, 12, 13, 14, 15, 28, 29, 30, 31, 32, 33, 34, 35, 36,  // Results = #37
/* 42001*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42004*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42007*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42010*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 42022*/   OPC_EmitInteger, MVT::i32, 1, 
/* 42025*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 42028*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42031*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42034*/   OPC_MorphNodeTo1, TARGET_VAL(R600::EXP_IEEE_eg), 0,
                 MVT::f32, 13/*#Ops*/, 2, 3, 4, 5, 37, 38, 39, 40, 41, 42, 43, 44, 45, 
             // Src: (fpow:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1) - Complexity = 3
             // Dst: (EXP_IEEE_eg:{ *:[f32] } (MUL:{ *:[i32] } f32:{ *:[f32] }:$src1, (LOG_IEEE_eg:{ *:[i32] } f32:{ *:[f32] }:$src0)))
/* 42053*/  /*Scope*/ 100|128,1/*228*/, /*->42283*/
/* 42055*/   OPC_CheckPatternPredicate, 3, // (Subtarget->hasCaymanISA())
/* 42057*/   OPC_EmitInteger, MVT::i32, 1, 
/* 42060*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42063*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42066*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42069*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42072*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42075*/   OPC_EmitInteger, MVT::i32, 1, 
/* 42078*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42081*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42084*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42087*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42090*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42093*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42096*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 42108*/   OPC_EmitInteger, MVT::i32, 1, 
/* 42111*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42114*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42117*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42120*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42123*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42126*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42129*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 42141*/   OPC_EmitInteger, MVT::i32, 1, 
/* 42144*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 42147*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42150*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42153*/   OPC_EmitNode1, TARGET_VAL(R600::LOG_IEEE_cm), 0,
                 MVT::i32, 13/*#Ops*/, 16, 17, 18, 19, 0, 20, 21, 22, 23, 24, 25, 26, 27,  // Results = #28
/* 42172*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42175*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42178*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42181*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 42193*/   OPC_EmitInteger, MVT::i32, 1, 
/* 42196*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 42199*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42202*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42205*/   OPC_EmitNode1, TARGET_VAL(R600::MUL), 0,
                 MVT::i32, 20/*#Ops*/, 6, 7, 8, 9, 10, 11, 1, 12, 13, 14, 15, 28, 29, 30, 31, 32, 33, 34, 35, 36,  // Results = #37
/* 42231*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42234*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42237*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42240*/   OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 42252*/   OPC_EmitInteger, MVT::i32, 1, 
/* 42255*/   OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 42258*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42261*/   OPC_EmitInteger, MVT::i32, 0, 
/* 42264*/   OPC_MorphNodeTo1, TARGET_VAL(R600::EXP_IEEE_cm), 0,
                 MVT::f32, 13/*#Ops*/, 2, 3, 4, 5, 37, 38, 39, 40, 41, 42, 43, 44, 45, 
             // Src: (fpow:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1) - Complexity = 3
             // Dst: (EXP_IEEE_cm:{ *:[f32] } (MUL:{ *:[i32] } f32:{ *:[f32] }:$src1, (LOG_IEEE_cm:{ *:[i32] } f32:{ *:[f32] }:$src0)))
/* 42283*/  0, /*End of Scope*/
/* 42284*/ /*SwitchOpcode*/ 14, TARGET_VAL(ISD::ConstantFP),// ->42301
/* 42287*/  OPC_RecordNode, // #0 = $val
/* 42288*/  OPC_CheckType, MVT::f32,
/* 42290*/  OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 42292*/  OPC_EmitConvertToTarget, 0,
/* 42294*/  OPC_MorphNodeTo1, TARGET_VAL(R600::MOV_IMM_F32), 0,
                MVT::f32, 1/*#Ops*/, 1, 
            // Src: (fpimm:{ *:[f32] }):$val - Complexity = 3
            // Dst: (MOV_IMM_F32:{ *:[f32] } (fpimm:{ *:[f32] }):$val)
/* 42301*/ /*SwitchOpcode*/ 10, TARGET_VAL(ISD::FABS),// ->42314
/* 42304*/  OPC_RecordChild0, // #0 = $src0
/* 42305*/  OPC_CheckType, MVT::f32,
/* 42307*/  OPC_MorphNodeTo1, TARGET_VAL(R600::FABS_R600), 0,
                MVT::f32, 1/*#Ops*/, 0, 
            // Src: (fabs:{ *:[f32] } f32:{ *:[f32] }:$src0) - Complexity = 3
            // Dst: (FABS_R600:{ *:[f32] } f32:{ *:[f32] }:$src0)
/* 42314*/ /*SwitchOpcode*/ 10, TARGET_VAL(ISD::FNEG),// ->42327
/* 42317*/  OPC_RecordChild0, // #0 = $src0
/* 42318*/  OPC_CheckType, MVT::f32,
/* 42320*/  OPC_MorphNodeTo1, TARGET_VAL(R600::FNEG_R600), 0,
                MVT::f32, 1/*#Ops*/, 0, 
            // Src: (fneg:{ *:[f32] } f32:{ *:[f32] }:$src0) - Complexity = 3
            // Dst: (FNEG_R600:{ *:[f32] } f32:{ *:[f32] }:$src0)
/* 42327*/ /*SwitchOpcode*/ 68|128,4/*580*/, TARGET_VAL(ISD::FCOPYSIGN),// ->42911
/* 42331*/  OPC_RecordChild0, // #0 = $src0
/* 42332*/  OPC_RecordChild1, // #1 = $src1
/* 42333*/  OPC_Scope, 19|128,2/*275*/, /*->42611*/ // 2 children in Scope
/* 42336*/   OPC_CheckChild1Type, MVT::f32,
/* 42338*/   OPC_SwitchType /*2 cases */, 112, MVT::f32,// ->42453
/* 42341*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 42343*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42346*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42349*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,7/*2147483647*/, 
/* 42356*/    OPC_EmitNode1, TARGET_VAL(R600::MOV_IMM_I32), 0,
                  MVT::i32, 1/*#Ops*/, 4,  // Results = #5
/* 42363*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42366*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42369*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 42381*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42384*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42387*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 42399*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42402*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42405*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 42417*/    OPC_EmitInteger, MVT::i32, 1, 
/* 42420*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 42423*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42426*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42429*/    OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                  MVT::f32, 18/*#Ops*/, 2, 3, 5, 6, 7, 8, 0, 9, 10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 
              // Src: (fcopysign:{ *:[f32] } f32:{ *:[f32] }:$src0, f32:{ *:[f32] }:$src1) - Complexity = 3
              // Dst: (BFI_INT_eg:{ *:[f32] } (MOV_IMM_I32:{ *:[i32] } 2147483647:{ *:[i32] }), ?:{ *:[f32] }:$src0, ?:{ *:[f32] }:$src1)
/* 42453*/   /*SwitchType*/ 26|128,1/*154*/, MVT::f64,// ->42610
/* 42456*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 42458*/    OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 42461*/    OPC_EmitInteger, MVT::i32, R600::sub0,
/* 42464*/    OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                  MVT::i32, 2/*#Ops*/, 0, 3,  // Results = #4
/* 42472*/    OPC_EmitInteger, MVT::i32, R600::sub0,
/* 42475*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42478*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42481*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,7/*2147483647*/, 
/* 42488*/    OPC_EmitNode1, TARGET_VAL(R600::MOV_IMM_I32), 0,
                  MVT::i32, 1/*#Ops*/, 8,  // Results = #9
/* 42495*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42498*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42501*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 42513*/    OPC_EmitInteger, MVT::i32, R600::sub1,
/* 42516*/    OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                  MVT::i32, 2/*#Ops*/, 0, 13,  // Results = #14
/* 42524*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42527*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42530*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 42542*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42545*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42548*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 42560*/    OPC_EmitInteger, MVT::i32, 1, 
/* 42563*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 42566*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42569*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42572*/    OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                  MVT::i32, 18/*#Ops*/, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 1, 18, 19, 20, 21, 22, 23, 24,  // Results = #25
/* 42596*/    OPC_EmitInteger, MVT::i32, R600::sub1,
/* 42599*/    OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                  MVT::f64, 5/*#Ops*/, 2, 4, 5, 25, 26, 
              // Src: (fcopysign:{ *:[f64] } f64:{ *:[f64] }:$src0, f32:{ *:[f32] }:$src1) - Complexity = 3
              // Dst: (REG_SEQUENCE:{ *:[f64] } R600_Reg64:{ *:[i32] }, (EXTRACT_SUBREG:{ *:[i32] } ?:{ *:[f64] }:$src0, sub0:{ *:[i32] }), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (MOV_IMM_I32:{ *:[i32] } 2147483647:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[f64] }:$src0, sub1:{ *:[i32] }), ?:{ *:[f32] }:$src1), sub1:{ *:[i32] })
/* 42610*/   0, // EndSwitchType
/* 42611*/  /*Scope*/ 41|128,2/*297*/, /*->42910*/
/* 42613*/   OPC_CheckChild1Type, MVT::f64,
/* 42615*/   OPC_SwitchType /*2 cases */, 123, MVT::f32,// ->42741
/* 42618*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 42620*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42623*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42626*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,7/*2147483647*/, 
/* 42633*/    OPC_EmitNode1, TARGET_VAL(R600::MOV_IMM_I32), 0,
                  MVT::i32, 1/*#Ops*/, 4,  // Results = #5
/* 42640*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42643*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42646*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 42658*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42661*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42664*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 42676*/    OPC_EmitInteger, MVT::i32, R600::sub1,
/* 42679*/    OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                  MVT::i32, 2/*#Ops*/, 1, 12,  // Results = #13
/* 42687*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42690*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42693*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 42705*/    OPC_EmitInteger, MVT::i32, 1, 
/* 42708*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 42711*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42714*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42717*/    OPC_MorphNodeTo1, TARGET_VAL(R600::BFI_INT_eg), 0,
                  MVT::f32, 18/*#Ops*/, 2, 3, 5, 6, 7, 8, 0, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 
              // Src: (fcopysign:{ *:[f32] } f32:{ *:[f32] }:$src0, f64:{ *:[f64] }:$src1) - Complexity = 3
              // Dst: (BFI_INT_eg:{ *:[f32] } (MOV_IMM_I32:{ *:[i32] } 2147483647:{ *:[i32] }), ?:{ *:[f32] }:$src0, (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[f64] }:$src1, sub1:{ *:[i32] }))
/* 42741*/   /*SwitchType*/ 37|128,1/*165*/, MVT::f64,// ->42909
/* 42744*/    OPC_CheckPatternPredicate, 0, // (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 42746*/    OPC_EmitInteger, MVT::i32, R600::R600_Reg64RegClassID,
/* 42749*/    OPC_EmitInteger, MVT::i32, R600::sub0,
/* 42752*/    OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                  MVT::i32, 2/*#Ops*/, 0, 3,  // Results = #4
/* 42760*/    OPC_EmitInteger, MVT::i32, R600::sub0,
/* 42763*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42766*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42769*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,7/*2147483647*/, 
/* 42776*/    OPC_EmitNode1, TARGET_VAL(R600::MOV_IMM_I32), 0,
                  MVT::i32, 1/*#Ops*/, 8,  // Results = #9
/* 42783*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42786*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42789*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 42801*/    OPC_EmitInteger, MVT::i32, R600::sub1,
/* 42804*/    OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                  MVT::i32, 2/*#Ops*/, 0, 13,  // Results = #14
/* 42812*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42815*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42818*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 42830*/    OPC_EmitInteger, MVT::i32, R600::sub1,
/* 42833*/    OPC_EmitNode1, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG), 0,
                  MVT::i32, 2/*#Ops*/, 1, 18,  // Results = #19
/* 42841*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42844*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42847*/    OPC_EmitInteger, MVT::i32, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/* 42859*/    OPC_EmitInteger, MVT::i32, 1, 
/* 42862*/    OPC_EmitRegister, MVT::i32, R600::PRED_SEL_OFF,
/* 42865*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42868*/    OPC_EmitInteger, MVT::i32, 0, 
/* 42871*/    OPC_EmitNode1, TARGET_VAL(R600::BFI_INT_eg), 0,
                  MVT::i32, 18/*#Ops*/, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26,  // Results = #27
/* 42895*/    OPC_EmitInteger, MVT::i32, R600::sub1,
/* 42898*/    OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::REG_SEQUENCE), 0,
                  MVT::f64, 5/*#Ops*/, 2, 4, 5, 27, 28, 
              // Src: (fcopysign:{ *:[f64] } f64:{ *:[f64] }:$src0, f64:{ *:[f64] }:$src1) - Complexity = 3
              // Dst: (REG_SEQUENCE:{ *:[f64] } R600_Reg64:{ *:[i32] }, (EXTRACT_SUBREG:{ *:[i32] } ?:{ *:[f64] }:$src0, sub0:{ *:[i32] }), sub0:{ *:[i32] }, (BFI_INT_eg:{ *:[i32] } (MOV_IMM_I32:{ *:[i32] } 2147483647:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[f64] }:$src0, sub1:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } R600_Reg64:{ *:[f64] }:$src1, sub1:{ *:[i32] })), sub1:{ *:[i32] })
/* 42909*/   0, // EndSwitchType
/* 42910*/  0, /*End of Scope*/
/* 42911*/ /*SwitchOpcode*/ 36|128,2/*292*/, TARGET_VAL(ISD::INSERT_VECTOR_ELT),// ->43207
/* 42915*/  OPC_RecordChild0, // #0 = $vec
/* 42916*/  OPC_RecordChild1, // #1 = $elem
/* 42917*/  OPC_Scope, 36, /*->42955*/ // 10 children in Scope
/* 42919*/   OPC_CheckChild2Integer, 0, 
/* 42921*/   OPC_SwitchType /*2 cases */, 14, MVT::v4i32,// ->42938
/* 42924*/    OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 42926*/    OPC_EmitInteger, MVT::i32, R600::sub0,
/* 42929*/    OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                  MVT::v4i32, 3/*#Ops*/, 0, 1, 2, 
              // Src: (insertelt:{ *:[v4i32] } v4i32:{ *:[v4i32] }:$vec, i32:{ *:[i32] }:$elem, 0:{ *:[iPTR] }) - Complexity = 8
              // Dst: (INSERT_SUBREG:{ *:[v4i32] } ?:{ *:[v4i32] }:$vec, ?:{ *:[i32] }:$elem, sub0:{ *:[i32] })
/* 42938*/   /*SwitchType*/ 14, MVT::v2i32,// ->42954
/* 42940*/    OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 42942*/    OPC_EmitInteger, MVT::i32, R600::sub0,
/* 42945*/    OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                  MVT::v2i32, 3/*#Ops*/, 0, 1, 2, 
              // Src: (insertelt:{ *:[v2i32] } v2i32:{ *:[v2i32] }:$vec, i32:{ *:[i32] }:$elem, 0:{ *:[iPTR] }) - Complexity = 8
              // Dst: (INSERT_SUBREG:{ *:[v2i32] } ?:{ *:[v2i32] }:$vec, ?:{ *:[i32] }:$elem, sub0:{ *:[i32] })
/* 42954*/   0, // EndSwitchType
/* 42955*/  /*Scope*/ 36, /*->42992*/
/* 42956*/   OPC_CheckChild2Integer, 1, 
/* 42958*/   OPC_SwitchType /*2 cases */, 14, MVT::v4i32,// ->42975
/* 42961*/    OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 42963*/    OPC_EmitInteger, MVT::i32, R600::sub1,
/* 42966*/    OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                  MVT::v4i32, 3/*#Ops*/, 0, 1, 2, 
              // Src: (insertelt:{ *:[v4i32] } v4i32:{ *:[v4i32] }:$vec, i32:{ *:[i32] }:$elem, 1:{ *:[iPTR] }) - Complexity = 8
              // Dst: (INSERT_SUBREG:{ *:[v4i32] } ?:{ *:[v4i32] }:$vec, ?:{ *:[i32] }:$elem, sub1:{ *:[i32] })
/* 42975*/   /*SwitchType*/ 14, MVT::v2i32,// ->42991
/* 42977*/    OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 42979*/    OPC_EmitInteger, MVT::i32, R600::sub1,
/* 42982*/    OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                  MVT::v2i32, 3/*#Ops*/, 0, 1, 2, 
              // Src: (insertelt:{ *:[v2i32] } v2i32:{ *:[v2i32] }:$vec, i32:{ *:[i32] }:$elem, 1:{ *:[iPTR] }) - Complexity = 8
              // Dst: (INSERT_SUBREG:{ *:[v2i32] } ?:{ *:[v2i32] }:$vec, ?:{ *:[i32] }:$elem, sub1:{ *:[i32] })
/* 42991*/   0, // EndSwitchType
/* 42992*/  /*Scope*/ 18, /*->43011*/
/* 42993*/   OPC_CheckChild2Integer, 2, 
/* 42995*/   OPC_CheckType, MVT::v4i32,
/* 42997*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 42999*/   OPC_EmitInteger, MVT::i32, R600::sub2,
/* 43002*/   OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                 MVT::v4i32, 3/*#Ops*/, 0, 1, 2, 
             // Src: (insertelt:{ *:[v4i32] } v4i32:{ *:[v4i32] }:$vec, i32:{ *:[i32] }:$elem, 2:{ *:[iPTR] }) - Complexity = 8
             // Dst: (INSERT_SUBREG:{ *:[v4i32] } ?:{ *:[v4i32] }:$vec, ?:{ *:[i32] }:$elem, sub2:{ *:[i32] })
/* 43011*/  /*Scope*/ 18, /*->43030*/
/* 43012*/   OPC_CheckChild2Integer, 3, 
/* 43014*/   OPC_CheckType, MVT::v4i32,
/* 43016*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 43018*/   OPC_EmitInteger, MVT::i32, R600::sub3,
/* 43021*/   OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                 MVT::v4i32, 3/*#Ops*/, 0, 1, 2, 
             // Src: (insertelt:{ *:[v4i32] } v4i32:{ *:[v4i32] }:$vec, i32:{ *:[i32] }:$elem, 3:{ *:[iPTR] }) - Complexity = 8
             // Dst: (INSERT_SUBREG:{ *:[v4i32] } ?:{ *:[v4i32] }:$vec, ?:{ *:[i32] }:$elem, sub3:{ *:[i32] })
/* 43030*/  /*Scope*/ 31, /*->43062*/
/* 43031*/   OPC_RecordChild2, // #2 = $index
/* 43032*/   OPC_CheckChild2Type, MVT::i32,
/* 43034*/   OPC_SwitchType /*2 cases */, 11, MVT::v2i32,// ->43048
/* 43037*/    OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 43039*/    OPC_MorphNodeTo1, TARGET_VAL(R600::R600_INSERT_ELT_V2), 0,
                  MVT::v2i32, 3/*#Ops*/, 0, 1, 2, 
              // Src: (insertelt:{ *:[v2i32] } v2i32:{ *:[v2i32] }:$vec, i32:{ *:[i32] }:$value, i32:{ *:[i32] }:$index) - Complexity = 3
              // Dst: (R600_INSERT_ELT_V2:{ *:[v2i32] } ?:{ *:[v2i32] }:$vec, ?:{ *:[i32] }:$value, ?:{ *:[i32] }:$index)
/* 43048*/   /*SwitchType*/ 11, MVT::v4i32,// ->43061
/* 43050*/    OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 43052*/    OPC_MorphNodeTo1, TARGET_VAL(R600::R600_INSERT_ELT_V4), 0,
                  MVT::v4i32, 3/*#Ops*/, 0, 1, 2, 
              // Src: (insertelt:{ *:[v4i32] } v4i32:{ *:[v4i32] }:$vec, i32:{ *:[i32] }:$value, i32:{ *:[i32] }:$index) - Complexity = 3
              // Dst: (R600_INSERT_ELT_V4:{ *:[v4i32] } ?:{ *:[v4i32] }:$vec, ?:{ *:[i32] }:$value, ?:{ *:[i32] }:$index)
/* 43061*/   0, // EndSwitchType
/* 43062*/  /*Scope*/ 36, /*->43099*/
/* 43063*/   OPC_CheckChild2Integer, 0, 
/* 43065*/   OPC_SwitchType /*2 cases */, 14, MVT::v4f32,// ->43082
/* 43068*/    OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 43070*/    OPC_EmitInteger, MVT::i32, R600::sub0,
/* 43073*/    OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                  MVT::v4f32, 3/*#Ops*/, 0, 1, 2, 
              // Src: (insertelt:{ *:[v4f32] } v4f32:{ *:[v4f32] }:$vec, f32:{ *:[f32] }:$elem, 0:{ *:[iPTR] }) - Complexity = 8
              // Dst: (INSERT_SUBREG:{ *:[v4f32] } ?:{ *:[v4f32] }:$vec, ?:{ *:[f32] }:$elem, sub0:{ *:[i32] })
/* 43082*/   /*SwitchType*/ 14, MVT::v2f32,// ->43098
/* 43084*/    OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 43086*/    OPC_EmitInteger, MVT::i32, R600::sub0,
/* 43089*/    OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                  MVT::v2f32, 3/*#Ops*/, 0, 1, 2, 
              // Src: (insertelt:{ *:[v2f32] } v2f32:{ *:[v2f32] }:$vec, f32:{ *:[f32] }:$elem, 0:{ *:[iPTR] }) - Complexity = 8
              // Dst: (INSERT_SUBREG:{ *:[v2f32] } ?:{ *:[v2f32] }:$vec, ?:{ *:[f32] }:$elem, sub0:{ *:[i32] })
/* 43098*/   0, // EndSwitchType
/* 43099*/  /*Scope*/ 36, /*->43136*/
/* 43100*/   OPC_CheckChild2Integer, 1, 
/* 43102*/   OPC_SwitchType /*2 cases */, 14, MVT::v4f32,// ->43119
/* 43105*/    OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 43107*/    OPC_EmitInteger, MVT::i32, R600::sub1,
/* 43110*/    OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                  MVT::v4f32, 3/*#Ops*/, 0, 1, 2, 
              // Src: (insertelt:{ *:[v4f32] } v4f32:{ *:[v4f32] }:$vec, f32:{ *:[f32] }:$elem, 1:{ *:[iPTR] }) - Complexity = 8
              // Dst: (INSERT_SUBREG:{ *:[v4f32] } ?:{ *:[v4f32] }:$vec, ?:{ *:[f32] }:$elem, sub1:{ *:[i32] })
/* 43119*/   /*SwitchType*/ 14, MVT::v2f32,// ->43135
/* 43121*/    OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 43123*/    OPC_EmitInteger, MVT::i32, R600::sub1,
/* 43126*/    OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                  MVT::v2f32, 3/*#Ops*/, 0, 1, 2, 
              // Src: (insertelt:{ *:[v2f32] } v2f32:{ *:[v2f32] }:$vec, f32:{ *:[f32] }:$elem, 1:{ *:[iPTR] }) - Complexity = 8
              // Dst: (INSERT_SUBREG:{ *:[v2f32] } ?:{ *:[v2f32] }:$vec, ?:{ *:[f32] }:$elem, sub1:{ *:[i32] })
/* 43135*/   0, // EndSwitchType
/* 43136*/  /*Scope*/ 18, /*->43155*/
/* 43137*/   OPC_CheckChild2Integer, 2, 
/* 43139*/   OPC_CheckType, MVT::v4f32,
/* 43141*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 43143*/   OPC_EmitInteger, MVT::i32, R600::sub2,
/* 43146*/   OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                 MVT::v4f32, 3/*#Ops*/, 0, 1, 2, 
             // Src: (insertelt:{ *:[v4f32] } v4f32:{ *:[v4f32] }:$vec, f32:{ *:[f32] }:$elem, 2:{ *:[iPTR] }) - Complexity = 8
             // Dst: (INSERT_SUBREG:{ *:[v4f32] } ?:{ *:[v4f32] }:$vec, ?:{ *:[f32] }:$elem, sub2:{ *:[i32] })
/* 43155*/  /*Scope*/ 18, /*->43174*/
/* 43156*/   OPC_CheckChild2Integer, 3, 
/* 43158*/   OPC_CheckType, MVT::v4f32,
/* 43160*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 43162*/   OPC_EmitInteger, MVT::i32, R600::sub3,
/* 43165*/   OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::INSERT_SUBREG), 0,
                 MVT::v4f32, 3/*#Ops*/, 0, 1, 2, 
             // Src: (insertelt:{ *:[v4f32] } v4f32:{ *:[v4f32] }:$vec, f32:{ *:[f32] }:$elem, 3:{ *:[iPTR] }) - Complexity = 8
             // Dst: (INSERT_SUBREG:{ *:[v4f32] } ?:{ *:[v4f32] }:$vec, ?:{ *:[f32] }:$elem, sub3:{ *:[i32] })
/* 43174*/  /*Scope*/ 31, /*->43206*/
/* 43175*/   OPC_RecordChild2, // #2 = $index
/* 43176*/   OPC_CheckChild2Type, MVT::i32,
/* 43178*/   OPC_SwitchType /*2 cases */, 11, MVT::v2f32,// ->43192
/* 43181*/    OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 43183*/    OPC_MorphNodeTo1, TARGET_VAL(R600::R600_INSERT_ELT_V2), 0,
                  MVT::v2f32, 3/*#Ops*/, 0, 1, 2, 
              // Src: (insertelt:{ *:[v2f32] } v2f32:{ *:[v2f32] }:$vec, f32:{ *:[f32] }:$value, i32:{ *:[i32] }:$index) - Complexity = 3
              // Dst: (R600_INSERT_ELT_V2:{ *:[v2f32] } ?:{ *:[v2f32] }:$vec, ?:{ *:[f32] }:$value, ?:{ *:[i32] }:$index)
/* 43192*/   /*SwitchType*/ 11, MVT::v4f32,// ->43205
/* 43194*/    OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 43196*/    OPC_MorphNodeTo1, TARGET_VAL(R600::R600_INSERT_ELT_V4), 0,
                  MVT::v4f32, 3/*#Ops*/, 0, 1, 2, 
              // Src: (insertelt:{ *:[v4f32] } v4f32:{ *:[v4f32] }:$vec, f32:{ *:[f32] }:$value, i32:{ *:[i32] }:$index) - Complexity = 3
              // Dst: (R600_INSERT_ELT_V4:{ *:[v4f32] } ?:{ *:[v4f32] }:$vec, ?:{ *:[f32] }:$value, ?:{ *:[i32] }:$index)
/* 43205*/   0, // EndSwitchType
/* 43206*/  0, /*End of Scope*/
/* 43207*/ /*SwitchOpcode*/ 96|128,18/*2400*/, TARGET_VAL(AMDGPUISD::TEXTURE_FETCH),// ->45611
/* 43211*/  OPC_Scope, 88|128,1/*216*/, /*->43430*/ // 11 children in Scope
/* 43214*/   OPC_CheckChild0Integer, 0, 
/* 43216*/   OPC_CheckChild0Type, MVT::i32,
/* 43218*/   OPC_RecordChild1, // #0 = $SRC_GPR
/* 43219*/   OPC_CheckChild1Type, MVT::v4f32,
/* 43221*/   OPC_RecordChild2, // #1 = $srcx
/* 43222*/   OPC_MoveChild2,
/* 43223*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43226*/   OPC_CheckType, MVT::i32,
/* 43228*/   OPC_MoveParent,
/* 43229*/   OPC_RecordChild3, // #2 = $srcy
/* 43230*/   OPC_MoveChild3,
/* 43231*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43234*/   OPC_CheckType, MVT::i32,
/* 43236*/   OPC_MoveParent,
/* 43237*/   OPC_RecordChild4, // #3 = $srcz
/* 43238*/   OPC_MoveChild4,
/* 43239*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43242*/   OPC_CheckType, MVT::i32,
/* 43244*/   OPC_MoveParent,
/* 43245*/   OPC_RecordChild5, // #4 = $srcw
/* 43246*/   OPC_MoveChild5,
/* 43247*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43250*/   OPC_CheckType, MVT::i32,
/* 43252*/   OPC_MoveParent,
/* 43253*/   OPC_RecordChild6, // #5 = $offsetx
/* 43254*/   OPC_MoveChild6,
/* 43255*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43258*/   OPC_CheckType, MVT::i32,
/* 43260*/   OPC_MoveParent,
/* 43261*/   OPC_RecordChild7, // #6 = $offsety
/* 43262*/   OPC_MoveChild7,
/* 43263*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43266*/   OPC_CheckType, MVT::i32,
/* 43268*/   OPC_MoveParent,
/* 43269*/   OPC_MoveChild, 8,
/* 43271*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43274*/   OPC_RecordNode, // #7 = $offsetz
/* 43275*/   OPC_CheckType, MVT::i32,
/* 43277*/   OPC_MoveParent,
/* 43278*/   OPC_MoveChild, 9,
/* 43280*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43283*/   OPC_RecordNode, // #8 = $DST_SEL_X
/* 43284*/   OPC_CheckType, MVT::i32,
/* 43286*/   OPC_MoveParent,
/* 43287*/   OPC_MoveChild, 10,
/* 43289*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43292*/   OPC_RecordNode, // #9 = $DST_SEL_Y
/* 43293*/   OPC_CheckType, MVT::i32,
/* 43295*/   OPC_MoveParent,
/* 43296*/   OPC_MoveChild, 11,
/* 43298*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43301*/   OPC_RecordNode, // #10 = $DST_SEL_Z
/* 43302*/   OPC_CheckType, MVT::i32,
/* 43304*/   OPC_MoveParent,
/* 43305*/   OPC_MoveChild, 12,
/* 43307*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43310*/   OPC_RecordNode, // #11 = $DST_SEL_W
/* 43311*/   OPC_CheckType, MVT::i32,
/* 43313*/   OPC_MoveParent,
/* 43314*/   OPC_MoveChild, 13,
/* 43316*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43319*/   OPC_RecordNode, // #12 = $RESOURCE_ID
/* 43320*/   OPC_CheckType, MVT::i32,
/* 43322*/   OPC_MoveParent,
/* 43323*/   OPC_MoveChild, 14,
/* 43325*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43328*/   OPC_RecordNode, // #13 = $SAMPLER_ID
/* 43329*/   OPC_CheckType, MVT::i32,
/* 43331*/   OPC_MoveParent,
/* 43332*/   OPC_MoveChild, 15,
/* 43334*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43337*/   OPC_RecordNode, // #14 = $COORD_TYPE_X
/* 43338*/   OPC_CheckType, MVT::i32,
/* 43340*/   OPC_MoveParent,
/* 43341*/   OPC_MoveChild, 16,
/* 43343*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43346*/   OPC_RecordNode, // #15 = $COORD_TYPE_Y
/* 43347*/   OPC_CheckType, MVT::i32,
/* 43349*/   OPC_MoveParent,
/* 43350*/   OPC_MoveChild, 17,
/* 43352*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43355*/   OPC_RecordNode, // #16 = $COORD_TYPE_Z
/* 43356*/   OPC_CheckType, MVT::i32,
/* 43358*/   OPC_MoveParent,
/* 43359*/   OPC_MoveChild, 18,
/* 43361*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43364*/   OPC_RecordNode, // #17 = $COORD_TYPE_W
/* 43365*/   OPC_CheckType, MVT::i32,
/* 43367*/   OPC_MoveParent,
/* 43368*/   OPC_CheckType, MVT::v4f32,
/* 43370*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 43372*/   OPC_EmitConvertToTarget, 1,
/* 43374*/   OPC_EmitConvertToTarget, 2,
/* 43376*/   OPC_EmitConvertToTarget, 3,
/* 43378*/   OPC_EmitConvertToTarget, 4,
/* 43380*/   OPC_EmitConvertToTarget, 5,
/* 43382*/   OPC_EmitConvertToTarget, 6,
/* 43384*/   OPC_EmitConvertToTarget, 7,
/* 43386*/   OPC_EmitConvertToTarget, 8,
/* 43388*/   OPC_EmitConvertToTarget, 9,
/* 43390*/   OPC_EmitConvertToTarget, 10,
/* 43392*/   OPC_EmitConvertToTarget, 11,
/* 43394*/   OPC_EmitConvertToTarget, 12,
/* 43396*/   OPC_EmitConvertToTarget, 13,
/* 43398*/   OPC_EmitConvertToTarget, 14,
/* 43400*/   OPC_EmitConvertToTarget, 15,
/* 43402*/   OPC_EmitConvertToTarget, 16,
/* 43404*/   OPC_EmitConvertToTarget, 17,
/* 43406*/   OPC_MorphNodeTo1, TARGET_VAL(R600::TEX_SAMPLE), 0,
                 MVT::v4f32, 18/*#Ops*/, 0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
             // Src: (TEXTURE_FETCH:{ *:[v4f32] } 0:{ *:[i32] }, v4f32:{ *:[v4f32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W) - Complexity = 59
             // Dst: (TEX_SAMPLE:{ *:[v4f32] } R600_Reg128:{ *:[v4f32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W)
/* 43430*/  /*Scope*/ 88|128,1/*216*/, /*->43648*/
/* 43432*/   OPC_CheckChild0Integer, 1, 
/* 43434*/   OPC_CheckChild0Type, MVT::i32,
/* 43436*/   OPC_RecordChild1, // #0 = $SRC_GPR
/* 43437*/   OPC_CheckChild1Type, MVT::v4f32,
/* 43439*/   OPC_RecordChild2, // #1 = $srcx
/* 43440*/   OPC_MoveChild2,
/* 43441*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43444*/   OPC_CheckType, MVT::i32,
/* 43446*/   OPC_MoveParent,
/* 43447*/   OPC_RecordChild3, // #2 = $srcy
/* 43448*/   OPC_MoveChild3,
/* 43449*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43452*/   OPC_CheckType, MVT::i32,
/* 43454*/   OPC_MoveParent,
/* 43455*/   OPC_RecordChild4, // #3 = $srcz
/* 43456*/   OPC_MoveChild4,
/* 43457*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43460*/   OPC_CheckType, MVT::i32,
/* 43462*/   OPC_MoveParent,
/* 43463*/   OPC_RecordChild5, // #4 = $srcw
/* 43464*/   OPC_MoveChild5,
/* 43465*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43468*/   OPC_CheckType, MVT::i32,
/* 43470*/   OPC_MoveParent,
/* 43471*/   OPC_RecordChild6, // #5 = $offsetx
/* 43472*/   OPC_MoveChild6,
/* 43473*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43476*/   OPC_CheckType, MVT::i32,
/* 43478*/   OPC_MoveParent,
/* 43479*/   OPC_RecordChild7, // #6 = $offsety
/* 43480*/   OPC_MoveChild7,
/* 43481*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43484*/   OPC_CheckType, MVT::i32,
/* 43486*/   OPC_MoveParent,
/* 43487*/   OPC_MoveChild, 8,
/* 43489*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43492*/   OPC_RecordNode, // #7 = $offsetz
/* 43493*/   OPC_CheckType, MVT::i32,
/* 43495*/   OPC_MoveParent,
/* 43496*/   OPC_MoveChild, 9,
/* 43498*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43501*/   OPC_RecordNode, // #8 = $DST_SEL_X
/* 43502*/   OPC_CheckType, MVT::i32,
/* 43504*/   OPC_MoveParent,
/* 43505*/   OPC_MoveChild, 10,
/* 43507*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43510*/   OPC_RecordNode, // #9 = $DST_SEL_Y
/* 43511*/   OPC_CheckType, MVT::i32,
/* 43513*/   OPC_MoveParent,
/* 43514*/   OPC_MoveChild, 11,
/* 43516*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43519*/   OPC_RecordNode, // #10 = $DST_SEL_Z
/* 43520*/   OPC_CheckType, MVT::i32,
/* 43522*/   OPC_MoveParent,
/* 43523*/   OPC_MoveChild, 12,
/* 43525*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43528*/   OPC_RecordNode, // #11 = $DST_SEL_W
/* 43529*/   OPC_CheckType, MVT::i32,
/* 43531*/   OPC_MoveParent,
/* 43532*/   OPC_MoveChild, 13,
/* 43534*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43537*/   OPC_RecordNode, // #12 = $RESOURCE_ID
/* 43538*/   OPC_CheckType, MVT::i32,
/* 43540*/   OPC_MoveParent,
/* 43541*/   OPC_MoveChild, 14,
/* 43543*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43546*/   OPC_RecordNode, // #13 = $SAMPLER_ID
/* 43547*/   OPC_CheckType, MVT::i32,
/* 43549*/   OPC_MoveParent,
/* 43550*/   OPC_MoveChild, 15,
/* 43552*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43555*/   OPC_RecordNode, // #14 = $COORD_TYPE_X
/* 43556*/   OPC_CheckType, MVT::i32,
/* 43558*/   OPC_MoveParent,
/* 43559*/   OPC_MoveChild, 16,
/* 43561*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43564*/   OPC_RecordNode, // #15 = $COORD_TYPE_Y
/* 43565*/   OPC_CheckType, MVT::i32,
/* 43567*/   OPC_MoveParent,
/* 43568*/   OPC_MoveChild, 17,
/* 43570*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43573*/   OPC_RecordNode, // #16 = $COORD_TYPE_Z
/* 43574*/   OPC_CheckType, MVT::i32,
/* 43576*/   OPC_MoveParent,
/* 43577*/   OPC_MoveChild, 18,
/* 43579*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43582*/   OPC_RecordNode, // #17 = $COORD_TYPE_W
/* 43583*/   OPC_CheckType, MVT::i32,
/* 43585*/   OPC_MoveParent,
/* 43586*/   OPC_CheckType, MVT::v4f32,
/* 43588*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 43590*/   OPC_EmitConvertToTarget, 1,
/* 43592*/   OPC_EmitConvertToTarget, 2,
/* 43594*/   OPC_EmitConvertToTarget, 3,
/* 43596*/   OPC_EmitConvertToTarget, 4,
/* 43598*/   OPC_EmitConvertToTarget, 5,
/* 43600*/   OPC_EmitConvertToTarget, 6,
/* 43602*/   OPC_EmitConvertToTarget, 7,
/* 43604*/   OPC_EmitConvertToTarget, 8,
/* 43606*/   OPC_EmitConvertToTarget, 9,
/* 43608*/   OPC_EmitConvertToTarget, 10,
/* 43610*/   OPC_EmitConvertToTarget, 11,
/* 43612*/   OPC_EmitConvertToTarget, 12,
/* 43614*/   OPC_EmitConvertToTarget, 13,
/* 43616*/   OPC_EmitConvertToTarget, 14,
/* 43618*/   OPC_EmitConvertToTarget, 15,
/* 43620*/   OPC_EmitConvertToTarget, 16,
/* 43622*/   OPC_EmitConvertToTarget, 17,
/* 43624*/   OPC_MorphNodeTo1, TARGET_VAL(R600::TEX_SAMPLE_C), 0,
                 MVT::v4f32, 18/*#Ops*/, 0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
             // Src: (TEXTURE_FETCH:{ *:[v4f32] } 1:{ *:[i32] }, v4f32:{ *:[v4f32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W) - Complexity = 59
             // Dst: (TEX_SAMPLE_C:{ *:[v4f32] } R600_Reg128:{ *:[v4f32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W)
/* 43648*/  /*Scope*/ 88|128,1/*216*/, /*->43866*/
/* 43650*/   OPC_CheckChild0Integer, 2, 
/* 43652*/   OPC_CheckChild0Type, MVT::i32,
/* 43654*/   OPC_RecordChild1, // #0 = $SRC_GPR
/* 43655*/   OPC_CheckChild1Type, MVT::v4f32,
/* 43657*/   OPC_RecordChild2, // #1 = $srcx
/* 43658*/   OPC_MoveChild2,
/* 43659*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43662*/   OPC_CheckType, MVT::i32,
/* 43664*/   OPC_MoveParent,
/* 43665*/   OPC_RecordChild3, // #2 = $srcy
/* 43666*/   OPC_MoveChild3,
/* 43667*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43670*/   OPC_CheckType, MVT::i32,
/* 43672*/   OPC_MoveParent,
/* 43673*/   OPC_RecordChild4, // #3 = $srcz
/* 43674*/   OPC_MoveChild4,
/* 43675*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43678*/   OPC_CheckType, MVT::i32,
/* 43680*/   OPC_MoveParent,
/* 43681*/   OPC_RecordChild5, // #4 = $srcw
/* 43682*/   OPC_MoveChild5,
/* 43683*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43686*/   OPC_CheckType, MVT::i32,
/* 43688*/   OPC_MoveParent,
/* 43689*/   OPC_RecordChild6, // #5 = $offsetx
/* 43690*/   OPC_MoveChild6,
/* 43691*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43694*/   OPC_CheckType, MVT::i32,
/* 43696*/   OPC_MoveParent,
/* 43697*/   OPC_RecordChild7, // #6 = $offsety
/* 43698*/   OPC_MoveChild7,
/* 43699*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43702*/   OPC_CheckType, MVT::i32,
/* 43704*/   OPC_MoveParent,
/* 43705*/   OPC_MoveChild, 8,
/* 43707*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43710*/   OPC_RecordNode, // #7 = $offsetz
/* 43711*/   OPC_CheckType, MVT::i32,
/* 43713*/   OPC_MoveParent,
/* 43714*/   OPC_MoveChild, 9,
/* 43716*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43719*/   OPC_RecordNode, // #8 = $DST_SEL_X
/* 43720*/   OPC_CheckType, MVT::i32,
/* 43722*/   OPC_MoveParent,
/* 43723*/   OPC_MoveChild, 10,
/* 43725*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43728*/   OPC_RecordNode, // #9 = $DST_SEL_Y
/* 43729*/   OPC_CheckType, MVT::i32,
/* 43731*/   OPC_MoveParent,
/* 43732*/   OPC_MoveChild, 11,
/* 43734*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43737*/   OPC_RecordNode, // #10 = $DST_SEL_Z
/* 43738*/   OPC_CheckType, MVT::i32,
/* 43740*/   OPC_MoveParent,
/* 43741*/   OPC_MoveChild, 12,
/* 43743*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43746*/   OPC_RecordNode, // #11 = $DST_SEL_W
/* 43747*/   OPC_CheckType, MVT::i32,
/* 43749*/   OPC_MoveParent,
/* 43750*/   OPC_MoveChild, 13,
/* 43752*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43755*/   OPC_RecordNode, // #12 = $RESOURCE_ID
/* 43756*/   OPC_CheckType, MVT::i32,
/* 43758*/   OPC_MoveParent,
/* 43759*/   OPC_MoveChild, 14,
/* 43761*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43764*/   OPC_RecordNode, // #13 = $SAMPLER_ID
/* 43765*/   OPC_CheckType, MVT::i32,
/* 43767*/   OPC_MoveParent,
/* 43768*/   OPC_MoveChild, 15,
/* 43770*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43773*/   OPC_RecordNode, // #14 = $COORD_TYPE_X
/* 43774*/   OPC_CheckType, MVT::i32,
/* 43776*/   OPC_MoveParent,
/* 43777*/   OPC_MoveChild, 16,
/* 43779*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43782*/   OPC_RecordNode, // #15 = $COORD_TYPE_Y
/* 43783*/   OPC_CheckType, MVT::i32,
/* 43785*/   OPC_MoveParent,
/* 43786*/   OPC_MoveChild, 17,
/* 43788*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43791*/   OPC_RecordNode, // #16 = $COORD_TYPE_Z
/* 43792*/   OPC_CheckType, MVT::i32,
/* 43794*/   OPC_MoveParent,
/* 43795*/   OPC_MoveChild, 18,
/* 43797*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43800*/   OPC_RecordNode, // #17 = $COORD_TYPE_W
/* 43801*/   OPC_CheckType, MVT::i32,
/* 43803*/   OPC_MoveParent,
/* 43804*/   OPC_CheckType, MVT::v4f32,
/* 43806*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 43808*/   OPC_EmitConvertToTarget, 1,
/* 43810*/   OPC_EmitConvertToTarget, 2,
/* 43812*/   OPC_EmitConvertToTarget, 3,
/* 43814*/   OPC_EmitConvertToTarget, 4,
/* 43816*/   OPC_EmitConvertToTarget, 5,
/* 43818*/   OPC_EmitConvertToTarget, 6,
/* 43820*/   OPC_EmitConvertToTarget, 7,
/* 43822*/   OPC_EmitConvertToTarget, 8,
/* 43824*/   OPC_EmitConvertToTarget, 9,
/* 43826*/   OPC_EmitConvertToTarget, 10,
/* 43828*/   OPC_EmitConvertToTarget, 11,
/* 43830*/   OPC_EmitConvertToTarget, 12,
/* 43832*/   OPC_EmitConvertToTarget, 13,
/* 43834*/   OPC_EmitConvertToTarget, 14,
/* 43836*/   OPC_EmitConvertToTarget, 15,
/* 43838*/   OPC_EmitConvertToTarget, 16,
/* 43840*/   OPC_EmitConvertToTarget, 17,
/* 43842*/   OPC_MorphNodeTo1, TARGET_VAL(R600::TEX_SAMPLE_L), 0,
                 MVT::v4f32, 18/*#Ops*/, 0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
             // Src: (TEXTURE_FETCH:{ *:[v4f32] } 2:{ *:[i32] }, v4f32:{ *:[v4f32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W) - Complexity = 59
             // Dst: (TEX_SAMPLE_L:{ *:[v4f32] } R600_Reg128:{ *:[v4f32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W)
/* 43866*/  /*Scope*/ 88|128,1/*216*/, /*->44084*/
/* 43868*/   OPC_CheckChild0Integer, 3, 
/* 43870*/   OPC_CheckChild0Type, MVT::i32,
/* 43872*/   OPC_RecordChild1, // #0 = $SRC_GPR
/* 43873*/   OPC_CheckChild1Type, MVT::v4f32,
/* 43875*/   OPC_RecordChild2, // #1 = $srcx
/* 43876*/   OPC_MoveChild2,
/* 43877*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43880*/   OPC_CheckType, MVT::i32,
/* 43882*/   OPC_MoveParent,
/* 43883*/   OPC_RecordChild3, // #2 = $srcy
/* 43884*/   OPC_MoveChild3,
/* 43885*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43888*/   OPC_CheckType, MVT::i32,
/* 43890*/   OPC_MoveParent,
/* 43891*/   OPC_RecordChild4, // #3 = $srcz
/* 43892*/   OPC_MoveChild4,
/* 43893*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43896*/   OPC_CheckType, MVT::i32,
/* 43898*/   OPC_MoveParent,
/* 43899*/   OPC_RecordChild5, // #4 = $srcw
/* 43900*/   OPC_MoveChild5,
/* 43901*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43904*/   OPC_CheckType, MVT::i32,
/* 43906*/   OPC_MoveParent,
/* 43907*/   OPC_RecordChild6, // #5 = $offsetx
/* 43908*/   OPC_MoveChild6,
/* 43909*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43912*/   OPC_CheckType, MVT::i32,
/* 43914*/   OPC_MoveParent,
/* 43915*/   OPC_RecordChild7, // #6 = $offsety
/* 43916*/   OPC_MoveChild7,
/* 43917*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43920*/   OPC_CheckType, MVT::i32,
/* 43922*/   OPC_MoveParent,
/* 43923*/   OPC_MoveChild, 8,
/* 43925*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43928*/   OPC_RecordNode, // #7 = $offsetz
/* 43929*/   OPC_CheckType, MVT::i32,
/* 43931*/   OPC_MoveParent,
/* 43932*/   OPC_MoveChild, 9,
/* 43934*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43937*/   OPC_RecordNode, // #8 = $DST_SEL_X
/* 43938*/   OPC_CheckType, MVT::i32,
/* 43940*/   OPC_MoveParent,
/* 43941*/   OPC_MoveChild, 10,
/* 43943*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43946*/   OPC_RecordNode, // #9 = $DST_SEL_Y
/* 43947*/   OPC_CheckType, MVT::i32,
/* 43949*/   OPC_MoveParent,
/* 43950*/   OPC_MoveChild, 11,
/* 43952*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43955*/   OPC_RecordNode, // #10 = $DST_SEL_Z
/* 43956*/   OPC_CheckType, MVT::i32,
/* 43958*/   OPC_MoveParent,
/* 43959*/   OPC_MoveChild, 12,
/* 43961*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43964*/   OPC_RecordNode, // #11 = $DST_SEL_W
/* 43965*/   OPC_CheckType, MVT::i32,
/* 43967*/   OPC_MoveParent,
/* 43968*/   OPC_MoveChild, 13,
/* 43970*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43973*/   OPC_RecordNode, // #12 = $RESOURCE_ID
/* 43974*/   OPC_CheckType, MVT::i32,
/* 43976*/   OPC_MoveParent,
/* 43977*/   OPC_MoveChild, 14,
/* 43979*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43982*/   OPC_RecordNode, // #13 = $SAMPLER_ID
/* 43983*/   OPC_CheckType, MVT::i32,
/* 43985*/   OPC_MoveParent,
/* 43986*/   OPC_MoveChild, 15,
/* 43988*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 43991*/   OPC_RecordNode, // #14 = $COORD_TYPE_X
/* 43992*/   OPC_CheckType, MVT::i32,
/* 43994*/   OPC_MoveParent,
/* 43995*/   OPC_MoveChild, 16,
/* 43997*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44000*/   OPC_RecordNode, // #15 = $COORD_TYPE_Y
/* 44001*/   OPC_CheckType, MVT::i32,
/* 44003*/   OPC_MoveParent,
/* 44004*/   OPC_MoveChild, 17,
/* 44006*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44009*/   OPC_RecordNode, // #16 = $COORD_TYPE_Z
/* 44010*/   OPC_CheckType, MVT::i32,
/* 44012*/   OPC_MoveParent,
/* 44013*/   OPC_MoveChild, 18,
/* 44015*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44018*/   OPC_RecordNode, // #17 = $COORD_TYPE_W
/* 44019*/   OPC_CheckType, MVT::i32,
/* 44021*/   OPC_MoveParent,
/* 44022*/   OPC_CheckType, MVT::v4f32,
/* 44024*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 44026*/   OPC_EmitConvertToTarget, 1,
/* 44028*/   OPC_EmitConvertToTarget, 2,
/* 44030*/   OPC_EmitConvertToTarget, 3,
/* 44032*/   OPC_EmitConvertToTarget, 4,
/* 44034*/   OPC_EmitConvertToTarget, 5,
/* 44036*/   OPC_EmitConvertToTarget, 6,
/* 44038*/   OPC_EmitConvertToTarget, 7,
/* 44040*/   OPC_EmitConvertToTarget, 8,
/* 44042*/   OPC_EmitConvertToTarget, 9,
/* 44044*/   OPC_EmitConvertToTarget, 10,
/* 44046*/   OPC_EmitConvertToTarget, 11,
/* 44048*/   OPC_EmitConvertToTarget, 12,
/* 44050*/   OPC_EmitConvertToTarget, 13,
/* 44052*/   OPC_EmitConvertToTarget, 14,
/* 44054*/   OPC_EmitConvertToTarget, 15,
/* 44056*/   OPC_EmitConvertToTarget, 16,
/* 44058*/   OPC_EmitConvertToTarget, 17,
/* 44060*/   OPC_MorphNodeTo1, TARGET_VAL(R600::TEX_SAMPLE_C_L), 0,
                 MVT::v4f32, 18/*#Ops*/, 0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
             // Src: (TEXTURE_FETCH:{ *:[v4f32] } 3:{ *:[i32] }, v4f32:{ *:[v4f32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W) - Complexity = 59
             // Dst: (TEX_SAMPLE_C_L:{ *:[v4f32] } R600_Reg128:{ *:[v4f32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W)
/* 44084*/  /*Scope*/ 88|128,1/*216*/, /*->44302*/
/* 44086*/   OPC_CheckChild0Integer, 4, 
/* 44088*/   OPC_CheckChild0Type, MVT::i32,
/* 44090*/   OPC_RecordChild1, // #0 = $SRC_GPR
/* 44091*/   OPC_CheckChild1Type, MVT::v4f32,
/* 44093*/   OPC_RecordChild2, // #1 = $srcx
/* 44094*/   OPC_MoveChild2,
/* 44095*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44098*/   OPC_CheckType, MVT::i32,
/* 44100*/   OPC_MoveParent,
/* 44101*/   OPC_RecordChild3, // #2 = $srcy
/* 44102*/   OPC_MoveChild3,
/* 44103*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44106*/   OPC_CheckType, MVT::i32,
/* 44108*/   OPC_MoveParent,
/* 44109*/   OPC_RecordChild4, // #3 = $srcz
/* 44110*/   OPC_MoveChild4,
/* 44111*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44114*/   OPC_CheckType, MVT::i32,
/* 44116*/   OPC_MoveParent,
/* 44117*/   OPC_RecordChild5, // #4 = $srcw
/* 44118*/   OPC_MoveChild5,
/* 44119*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44122*/   OPC_CheckType, MVT::i32,
/* 44124*/   OPC_MoveParent,
/* 44125*/   OPC_RecordChild6, // #5 = $offsetx
/* 44126*/   OPC_MoveChild6,
/* 44127*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44130*/   OPC_CheckType, MVT::i32,
/* 44132*/   OPC_MoveParent,
/* 44133*/   OPC_RecordChild7, // #6 = $offsety
/* 44134*/   OPC_MoveChild7,
/* 44135*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44138*/   OPC_CheckType, MVT::i32,
/* 44140*/   OPC_MoveParent,
/* 44141*/   OPC_MoveChild, 8,
/* 44143*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44146*/   OPC_RecordNode, // #7 = $offsetz
/* 44147*/   OPC_CheckType, MVT::i32,
/* 44149*/   OPC_MoveParent,
/* 44150*/   OPC_MoveChild, 9,
/* 44152*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44155*/   OPC_RecordNode, // #8 = $DST_SEL_X
/* 44156*/   OPC_CheckType, MVT::i32,
/* 44158*/   OPC_MoveParent,
/* 44159*/   OPC_MoveChild, 10,
/* 44161*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44164*/   OPC_RecordNode, // #9 = $DST_SEL_Y
/* 44165*/   OPC_CheckType, MVT::i32,
/* 44167*/   OPC_MoveParent,
/* 44168*/   OPC_MoveChild, 11,
/* 44170*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44173*/   OPC_RecordNode, // #10 = $DST_SEL_Z
/* 44174*/   OPC_CheckType, MVT::i32,
/* 44176*/   OPC_MoveParent,
/* 44177*/   OPC_MoveChild, 12,
/* 44179*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44182*/   OPC_RecordNode, // #11 = $DST_SEL_W
/* 44183*/   OPC_CheckType, MVT::i32,
/* 44185*/   OPC_MoveParent,
/* 44186*/   OPC_MoveChild, 13,
/* 44188*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44191*/   OPC_RecordNode, // #12 = $RESOURCE_ID
/* 44192*/   OPC_CheckType, MVT::i32,
/* 44194*/   OPC_MoveParent,
/* 44195*/   OPC_MoveChild, 14,
/* 44197*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44200*/   OPC_RecordNode, // #13 = $SAMPLER_ID
/* 44201*/   OPC_CheckType, MVT::i32,
/* 44203*/   OPC_MoveParent,
/* 44204*/   OPC_MoveChild, 15,
/* 44206*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44209*/   OPC_RecordNode, // #14 = $COORD_TYPE_X
/* 44210*/   OPC_CheckType, MVT::i32,
/* 44212*/   OPC_MoveParent,
/* 44213*/   OPC_MoveChild, 16,
/* 44215*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44218*/   OPC_RecordNode, // #15 = $COORD_TYPE_Y
/* 44219*/   OPC_CheckType, MVT::i32,
/* 44221*/   OPC_MoveParent,
/* 44222*/   OPC_MoveChild, 17,
/* 44224*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44227*/   OPC_RecordNode, // #16 = $COORD_TYPE_Z
/* 44228*/   OPC_CheckType, MVT::i32,
/* 44230*/   OPC_MoveParent,
/* 44231*/   OPC_MoveChild, 18,
/* 44233*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44236*/   OPC_RecordNode, // #17 = $COORD_TYPE_W
/* 44237*/   OPC_CheckType, MVT::i32,
/* 44239*/   OPC_MoveParent,
/* 44240*/   OPC_CheckType, MVT::v4f32,
/* 44242*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 44244*/   OPC_EmitConvertToTarget, 1,
/* 44246*/   OPC_EmitConvertToTarget, 2,
/* 44248*/   OPC_EmitConvertToTarget, 3,
/* 44250*/   OPC_EmitConvertToTarget, 4,
/* 44252*/   OPC_EmitConvertToTarget, 5,
/* 44254*/   OPC_EmitConvertToTarget, 6,
/* 44256*/   OPC_EmitConvertToTarget, 7,
/* 44258*/   OPC_EmitConvertToTarget, 8,
/* 44260*/   OPC_EmitConvertToTarget, 9,
/* 44262*/   OPC_EmitConvertToTarget, 10,
/* 44264*/   OPC_EmitConvertToTarget, 11,
/* 44266*/   OPC_EmitConvertToTarget, 12,
/* 44268*/   OPC_EmitConvertToTarget, 13,
/* 44270*/   OPC_EmitConvertToTarget, 14,
/* 44272*/   OPC_EmitConvertToTarget, 15,
/* 44274*/   OPC_EmitConvertToTarget, 16,
/* 44276*/   OPC_EmitConvertToTarget, 17,
/* 44278*/   OPC_MorphNodeTo1, TARGET_VAL(R600::TEX_SAMPLE_LB), 0,
                 MVT::v4f32, 18/*#Ops*/, 0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
             // Src: (TEXTURE_FETCH:{ *:[v4f32] } 4:{ *:[i32] }, v4f32:{ *:[v4f32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W) - Complexity = 59
             // Dst: (TEX_SAMPLE_LB:{ *:[v4f32] } R600_Reg128:{ *:[v4f32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W)
/* 44302*/  /*Scope*/ 88|128,1/*216*/, /*->44520*/
/* 44304*/   OPC_CheckChild0Integer, 5, 
/* 44306*/   OPC_CheckChild0Type, MVT::i32,
/* 44308*/   OPC_RecordChild1, // #0 = $SRC_GPR
/* 44309*/   OPC_CheckChild1Type, MVT::v4f32,
/* 44311*/   OPC_RecordChild2, // #1 = $srcx
/* 44312*/   OPC_MoveChild2,
/* 44313*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44316*/   OPC_CheckType, MVT::i32,
/* 44318*/   OPC_MoveParent,
/* 44319*/   OPC_RecordChild3, // #2 = $srcy
/* 44320*/   OPC_MoveChild3,
/* 44321*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44324*/   OPC_CheckType, MVT::i32,
/* 44326*/   OPC_MoveParent,
/* 44327*/   OPC_RecordChild4, // #3 = $srcz
/* 44328*/   OPC_MoveChild4,
/* 44329*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44332*/   OPC_CheckType, MVT::i32,
/* 44334*/   OPC_MoveParent,
/* 44335*/   OPC_RecordChild5, // #4 = $srcw
/* 44336*/   OPC_MoveChild5,
/* 44337*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44340*/   OPC_CheckType, MVT::i32,
/* 44342*/   OPC_MoveParent,
/* 44343*/   OPC_RecordChild6, // #5 = $offsetx
/* 44344*/   OPC_MoveChild6,
/* 44345*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44348*/   OPC_CheckType, MVT::i32,
/* 44350*/   OPC_MoveParent,
/* 44351*/   OPC_RecordChild7, // #6 = $offsety
/* 44352*/   OPC_MoveChild7,
/* 44353*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44356*/   OPC_CheckType, MVT::i32,
/* 44358*/   OPC_MoveParent,
/* 44359*/   OPC_MoveChild, 8,
/* 44361*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44364*/   OPC_RecordNode, // #7 = $offsetz
/* 44365*/   OPC_CheckType, MVT::i32,
/* 44367*/   OPC_MoveParent,
/* 44368*/   OPC_MoveChild, 9,
/* 44370*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44373*/   OPC_RecordNode, // #8 = $DST_SEL_X
/* 44374*/   OPC_CheckType, MVT::i32,
/* 44376*/   OPC_MoveParent,
/* 44377*/   OPC_MoveChild, 10,
/* 44379*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44382*/   OPC_RecordNode, // #9 = $DST_SEL_Y
/* 44383*/   OPC_CheckType, MVT::i32,
/* 44385*/   OPC_MoveParent,
/* 44386*/   OPC_MoveChild, 11,
/* 44388*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44391*/   OPC_RecordNode, // #10 = $DST_SEL_Z
/* 44392*/   OPC_CheckType, MVT::i32,
/* 44394*/   OPC_MoveParent,
/* 44395*/   OPC_MoveChild, 12,
/* 44397*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44400*/   OPC_RecordNode, // #11 = $DST_SEL_W
/* 44401*/   OPC_CheckType, MVT::i32,
/* 44403*/   OPC_MoveParent,
/* 44404*/   OPC_MoveChild, 13,
/* 44406*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44409*/   OPC_RecordNode, // #12 = $RESOURCE_ID
/* 44410*/   OPC_CheckType, MVT::i32,
/* 44412*/   OPC_MoveParent,
/* 44413*/   OPC_MoveChild, 14,
/* 44415*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44418*/   OPC_RecordNode, // #13 = $SAMPLER_ID
/* 44419*/   OPC_CheckType, MVT::i32,
/* 44421*/   OPC_MoveParent,
/* 44422*/   OPC_MoveChild, 15,
/* 44424*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44427*/   OPC_RecordNode, // #14 = $COORD_TYPE_X
/* 44428*/   OPC_CheckType, MVT::i32,
/* 44430*/   OPC_MoveParent,
/* 44431*/   OPC_MoveChild, 16,
/* 44433*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44436*/   OPC_RecordNode, // #15 = $COORD_TYPE_Y
/* 44437*/   OPC_CheckType, MVT::i32,
/* 44439*/   OPC_MoveParent,
/* 44440*/   OPC_MoveChild, 17,
/* 44442*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44445*/   OPC_RecordNode, // #16 = $COORD_TYPE_Z
/* 44446*/   OPC_CheckType, MVT::i32,
/* 44448*/   OPC_MoveParent,
/* 44449*/   OPC_MoveChild, 18,
/* 44451*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44454*/   OPC_RecordNode, // #17 = $COORD_TYPE_W
/* 44455*/   OPC_CheckType, MVT::i32,
/* 44457*/   OPC_MoveParent,
/* 44458*/   OPC_CheckType, MVT::v4f32,
/* 44460*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 44462*/   OPC_EmitConvertToTarget, 1,
/* 44464*/   OPC_EmitConvertToTarget, 2,
/* 44466*/   OPC_EmitConvertToTarget, 3,
/* 44468*/   OPC_EmitConvertToTarget, 4,
/* 44470*/   OPC_EmitConvertToTarget, 5,
/* 44472*/   OPC_EmitConvertToTarget, 6,
/* 44474*/   OPC_EmitConvertToTarget, 7,
/* 44476*/   OPC_EmitConvertToTarget, 8,
/* 44478*/   OPC_EmitConvertToTarget, 9,
/* 44480*/   OPC_EmitConvertToTarget, 10,
/* 44482*/   OPC_EmitConvertToTarget, 11,
/* 44484*/   OPC_EmitConvertToTarget, 12,
/* 44486*/   OPC_EmitConvertToTarget, 13,
/* 44488*/   OPC_EmitConvertToTarget, 14,
/* 44490*/   OPC_EmitConvertToTarget, 15,
/* 44492*/   OPC_EmitConvertToTarget, 16,
/* 44494*/   OPC_EmitConvertToTarget, 17,
/* 44496*/   OPC_MorphNodeTo1, TARGET_VAL(R600::TEX_SAMPLE_C_LB), 0,
                 MVT::v4f32, 18/*#Ops*/, 0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
             // Src: (TEXTURE_FETCH:{ *:[v4f32] } 5:{ *:[i32] }, v4f32:{ *:[v4f32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W) - Complexity = 59
             // Dst: (TEX_SAMPLE_C_LB:{ *:[v4f32] } R600_Reg128:{ *:[v4f32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W)
/* 44520*/  /*Scope*/ 88|128,1/*216*/, /*->44738*/
/* 44522*/   OPC_CheckChild0Integer, 6, 
/* 44524*/   OPC_CheckChild0Type, MVT::i32,
/* 44526*/   OPC_RecordChild1, // #0 = $SRC_GPR
/* 44527*/   OPC_CheckChild1Type, MVT::v4i32,
/* 44529*/   OPC_RecordChild2, // #1 = $srcx
/* 44530*/   OPC_MoveChild2,
/* 44531*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44534*/   OPC_CheckType, MVT::i32,
/* 44536*/   OPC_MoveParent,
/* 44537*/   OPC_RecordChild3, // #2 = $srcy
/* 44538*/   OPC_MoveChild3,
/* 44539*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44542*/   OPC_CheckType, MVT::i32,
/* 44544*/   OPC_MoveParent,
/* 44545*/   OPC_RecordChild4, // #3 = $srcz
/* 44546*/   OPC_MoveChild4,
/* 44547*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44550*/   OPC_CheckType, MVT::i32,
/* 44552*/   OPC_MoveParent,
/* 44553*/   OPC_RecordChild5, // #4 = $srcw
/* 44554*/   OPC_MoveChild5,
/* 44555*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44558*/   OPC_CheckType, MVT::i32,
/* 44560*/   OPC_MoveParent,
/* 44561*/   OPC_RecordChild6, // #5 = $offsetx
/* 44562*/   OPC_MoveChild6,
/* 44563*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44566*/   OPC_CheckType, MVT::i32,
/* 44568*/   OPC_MoveParent,
/* 44569*/   OPC_RecordChild7, // #6 = $offsety
/* 44570*/   OPC_MoveChild7,
/* 44571*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44574*/   OPC_CheckType, MVT::i32,
/* 44576*/   OPC_MoveParent,
/* 44577*/   OPC_MoveChild, 8,
/* 44579*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44582*/   OPC_RecordNode, // #7 = $offsetz
/* 44583*/   OPC_CheckType, MVT::i32,
/* 44585*/   OPC_MoveParent,
/* 44586*/   OPC_MoveChild, 9,
/* 44588*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44591*/   OPC_RecordNode, // #8 = $DST_SEL_X
/* 44592*/   OPC_CheckType, MVT::i32,
/* 44594*/   OPC_MoveParent,
/* 44595*/   OPC_MoveChild, 10,
/* 44597*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44600*/   OPC_RecordNode, // #9 = $DST_SEL_Y
/* 44601*/   OPC_CheckType, MVT::i32,
/* 44603*/   OPC_MoveParent,
/* 44604*/   OPC_MoveChild, 11,
/* 44606*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44609*/   OPC_RecordNode, // #10 = $DST_SEL_Z
/* 44610*/   OPC_CheckType, MVT::i32,
/* 44612*/   OPC_MoveParent,
/* 44613*/   OPC_MoveChild, 12,
/* 44615*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44618*/   OPC_RecordNode, // #11 = $DST_SEL_W
/* 44619*/   OPC_CheckType, MVT::i32,
/* 44621*/   OPC_MoveParent,
/* 44622*/   OPC_MoveChild, 13,
/* 44624*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44627*/   OPC_RecordNode, // #12 = $RESOURCE_ID
/* 44628*/   OPC_CheckType, MVT::i32,
/* 44630*/   OPC_MoveParent,
/* 44631*/   OPC_MoveChild, 14,
/* 44633*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44636*/   OPC_RecordNode, // #13 = $SAMPLER_ID
/* 44637*/   OPC_CheckType, MVT::i32,
/* 44639*/   OPC_MoveParent,
/* 44640*/   OPC_MoveChild, 15,
/* 44642*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44645*/   OPC_RecordNode, // #14 = $COORD_TYPE_X
/* 44646*/   OPC_CheckType, MVT::i32,
/* 44648*/   OPC_MoveParent,
/* 44649*/   OPC_MoveChild, 16,
/* 44651*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44654*/   OPC_RecordNode, // #15 = $COORD_TYPE_Y
/* 44655*/   OPC_CheckType, MVT::i32,
/* 44657*/   OPC_MoveParent,
/* 44658*/   OPC_MoveChild, 17,
/* 44660*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44663*/   OPC_RecordNode, // #16 = $COORD_TYPE_Z
/* 44664*/   OPC_CheckType, MVT::i32,
/* 44666*/   OPC_MoveParent,
/* 44667*/   OPC_MoveChild, 18,
/* 44669*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44672*/   OPC_RecordNode, // #17 = $COORD_TYPE_W
/* 44673*/   OPC_CheckType, MVT::i32,
/* 44675*/   OPC_MoveParent,
/* 44676*/   OPC_CheckType, MVT::v4f32,
/* 44678*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 44680*/   OPC_EmitConvertToTarget, 1,
/* 44682*/   OPC_EmitConvertToTarget, 2,
/* 44684*/   OPC_EmitConvertToTarget, 3,
/* 44686*/   OPC_EmitConvertToTarget, 4,
/* 44688*/   OPC_EmitConvertToTarget, 5,
/* 44690*/   OPC_EmitConvertToTarget, 6,
/* 44692*/   OPC_EmitConvertToTarget, 7,
/* 44694*/   OPC_EmitConvertToTarget, 8,
/* 44696*/   OPC_EmitConvertToTarget, 9,
/* 44698*/   OPC_EmitConvertToTarget, 10,
/* 44700*/   OPC_EmitConvertToTarget, 11,
/* 44702*/   OPC_EmitConvertToTarget, 12,
/* 44704*/   OPC_EmitConvertToTarget, 13,
/* 44706*/   OPC_EmitConvertToTarget, 14,
/* 44708*/   OPC_EmitConvertToTarget, 15,
/* 44710*/   OPC_EmitConvertToTarget, 16,
/* 44712*/   OPC_EmitConvertToTarget, 17,
/* 44714*/   OPC_MorphNodeTo1, TARGET_VAL(R600::TEX_LD), 0,
                 MVT::v4f32, 18/*#Ops*/, 0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
             // Src: (TEXTURE_FETCH:{ *:[v4f32] } 6:{ *:[i32] }, v4i32:{ *:[v4i32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W) - Complexity = 59
             // Dst: (TEX_LD:{ *:[v4f32] } R600_Reg128:{ *:[v4i32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W)
/* 44738*/  /*Scope*/ 88|128,1/*216*/, /*->44956*/
/* 44740*/   OPC_CheckChild0Integer, 7, 
/* 44742*/   OPC_CheckChild0Type, MVT::i32,
/* 44744*/   OPC_RecordChild1, // #0 = $SRC_GPR
/* 44745*/   OPC_CheckChild1Type, MVT::v4i32,
/* 44747*/   OPC_RecordChild2, // #1 = $srcx
/* 44748*/   OPC_MoveChild2,
/* 44749*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44752*/   OPC_CheckType, MVT::i32,
/* 44754*/   OPC_MoveParent,
/* 44755*/   OPC_RecordChild3, // #2 = $srcy
/* 44756*/   OPC_MoveChild3,
/* 44757*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44760*/   OPC_CheckType, MVT::i32,
/* 44762*/   OPC_MoveParent,
/* 44763*/   OPC_RecordChild4, // #3 = $srcz
/* 44764*/   OPC_MoveChild4,
/* 44765*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44768*/   OPC_CheckType, MVT::i32,
/* 44770*/   OPC_MoveParent,
/* 44771*/   OPC_RecordChild5, // #4 = $srcw
/* 44772*/   OPC_MoveChild5,
/* 44773*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44776*/   OPC_CheckType, MVT::i32,
/* 44778*/   OPC_MoveParent,
/* 44779*/   OPC_RecordChild6, // #5 = $offsetx
/* 44780*/   OPC_MoveChild6,
/* 44781*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44784*/   OPC_CheckType, MVT::i32,
/* 44786*/   OPC_MoveParent,
/* 44787*/   OPC_RecordChild7, // #6 = $offsety
/* 44788*/   OPC_MoveChild7,
/* 44789*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44792*/   OPC_CheckType, MVT::i32,
/* 44794*/   OPC_MoveParent,
/* 44795*/   OPC_MoveChild, 8,
/* 44797*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44800*/   OPC_RecordNode, // #7 = $offsetz
/* 44801*/   OPC_CheckType, MVT::i32,
/* 44803*/   OPC_MoveParent,
/* 44804*/   OPC_MoveChild, 9,
/* 44806*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44809*/   OPC_RecordNode, // #8 = $DST_SEL_X
/* 44810*/   OPC_CheckType, MVT::i32,
/* 44812*/   OPC_MoveParent,
/* 44813*/   OPC_MoveChild, 10,
/* 44815*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44818*/   OPC_RecordNode, // #9 = $DST_SEL_Y
/* 44819*/   OPC_CheckType, MVT::i32,
/* 44821*/   OPC_MoveParent,
/* 44822*/   OPC_MoveChild, 11,
/* 44824*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44827*/   OPC_RecordNode, // #10 = $DST_SEL_Z
/* 44828*/   OPC_CheckType, MVT::i32,
/* 44830*/   OPC_MoveParent,
/* 44831*/   OPC_MoveChild, 12,
/* 44833*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44836*/   OPC_RecordNode, // #11 = $DST_SEL_W
/* 44837*/   OPC_CheckType, MVT::i32,
/* 44839*/   OPC_MoveParent,
/* 44840*/   OPC_MoveChild, 13,
/* 44842*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44845*/   OPC_RecordNode, // #12 = $RESOURCE_ID
/* 44846*/   OPC_CheckType, MVT::i32,
/* 44848*/   OPC_MoveParent,
/* 44849*/   OPC_MoveChild, 14,
/* 44851*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44854*/   OPC_RecordNode, // #13 = $SAMPLER_ID
/* 44855*/   OPC_CheckType, MVT::i32,
/* 44857*/   OPC_MoveParent,
/* 44858*/   OPC_MoveChild, 15,
/* 44860*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44863*/   OPC_RecordNode, // #14 = $COORD_TYPE_X
/* 44864*/   OPC_CheckType, MVT::i32,
/* 44866*/   OPC_MoveParent,
/* 44867*/   OPC_MoveChild, 16,
/* 44869*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44872*/   OPC_RecordNode, // #15 = $COORD_TYPE_Y
/* 44873*/   OPC_CheckType, MVT::i32,
/* 44875*/   OPC_MoveParent,
/* 44876*/   OPC_MoveChild, 17,
/* 44878*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44881*/   OPC_RecordNode, // #16 = $COORD_TYPE_Z
/* 44882*/   OPC_CheckType, MVT::i32,
/* 44884*/   OPC_MoveParent,
/* 44885*/   OPC_MoveChild, 18,
/* 44887*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44890*/   OPC_RecordNode, // #17 = $COORD_TYPE_W
/* 44891*/   OPC_CheckType, MVT::i32,
/* 44893*/   OPC_MoveParent,
/* 44894*/   OPC_CheckType, MVT::v4f32,
/* 44896*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 44898*/   OPC_EmitConvertToTarget, 1,
/* 44900*/   OPC_EmitConvertToTarget, 2,
/* 44902*/   OPC_EmitConvertToTarget, 3,
/* 44904*/   OPC_EmitConvertToTarget, 4,
/* 44906*/   OPC_EmitConvertToTarget, 5,
/* 44908*/   OPC_EmitConvertToTarget, 6,
/* 44910*/   OPC_EmitConvertToTarget, 7,
/* 44912*/   OPC_EmitConvertToTarget, 8,
/* 44914*/   OPC_EmitConvertToTarget, 9,
/* 44916*/   OPC_EmitConvertToTarget, 10,
/* 44918*/   OPC_EmitConvertToTarget, 11,
/* 44920*/   OPC_EmitConvertToTarget, 12,
/* 44922*/   OPC_EmitConvertToTarget, 13,
/* 44924*/   OPC_EmitConvertToTarget, 14,
/* 44926*/   OPC_EmitConvertToTarget, 15,
/* 44928*/   OPC_EmitConvertToTarget, 16,
/* 44930*/   OPC_EmitConvertToTarget, 17,
/* 44932*/   OPC_MorphNodeTo1, TARGET_VAL(R600::TEX_GET_TEXTURE_RESINFO), 0,
                 MVT::v4f32, 18/*#Ops*/, 0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
             // Src: (TEXTURE_FETCH:{ *:[v4f32] } 7:{ *:[i32] }, v4i32:{ *:[v4i32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W) - Complexity = 59
             // Dst: (TEX_GET_TEXTURE_RESINFO:{ *:[v4f32] } R600_Reg128:{ *:[v4i32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W)
/* 44956*/  /*Scope*/ 88|128,1/*216*/, /*->45174*/
/* 44958*/   OPC_CheckChild0Integer, 8, 
/* 44960*/   OPC_CheckChild0Type, MVT::i32,
/* 44962*/   OPC_RecordChild1, // #0 = $SRC_GPR
/* 44963*/   OPC_CheckChild1Type, MVT::v4f32,
/* 44965*/   OPC_RecordChild2, // #1 = $srcx
/* 44966*/   OPC_MoveChild2,
/* 44967*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44970*/   OPC_CheckType, MVT::i32,
/* 44972*/   OPC_MoveParent,
/* 44973*/   OPC_RecordChild3, // #2 = $srcy
/* 44974*/   OPC_MoveChild3,
/* 44975*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44978*/   OPC_CheckType, MVT::i32,
/* 44980*/   OPC_MoveParent,
/* 44981*/   OPC_RecordChild4, // #3 = $srcz
/* 44982*/   OPC_MoveChild4,
/* 44983*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44986*/   OPC_CheckType, MVT::i32,
/* 44988*/   OPC_MoveParent,
/* 44989*/   OPC_RecordChild5, // #4 = $srcw
/* 44990*/   OPC_MoveChild5,
/* 44991*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 44994*/   OPC_CheckType, MVT::i32,
/* 44996*/   OPC_MoveParent,
/* 44997*/   OPC_RecordChild6, // #5 = $offsetx
/* 44998*/   OPC_MoveChild6,
/* 44999*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45002*/   OPC_CheckType, MVT::i32,
/* 45004*/   OPC_MoveParent,
/* 45005*/   OPC_RecordChild7, // #6 = $offsety
/* 45006*/   OPC_MoveChild7,
/* 45007*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45010*/   OPC_CheckType, MVT::i32,
/* 45012*/   OPC_MoveParent,
/* 45013*/   OPC_MoveChild, 8,
/* 45015*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45018*/   OPC_RecordNode, // #7 = $offsetz
/* 45019*/   OPC_CheckType, MVT::i32,
/* 45021*/   OPC_MoveParent,
/* 45022*/   OPC_MoveChild, 9,
/* 45024*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45027*/   OPC_RecordNode, // #8 = $DST_SEL_X
/* 45028*/   OPC_CheckType, MVT::i32,
/* 45030*/   OPC_MoveParent,
/* 45031*/   OPC_MoveChild, 10,
/* 45033*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45036*/   OPC_RecordNode, // #9 = $DST_SEL_Y
/* 45037*/   OPC_CheckType, MVT::i32,
/* 45039*/   OPC_MoveParent,
/* 45040*/   OPC_MoveChild, 11,
/* 45042*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45045*/   OPC_RecordNode, // #10 = $DST_SEL_Z
/* 45046*/   OPC_CheckType, MVT::i32,
/* 45048*/   OPC_MoveParent,
/* 45049*/   OPC_MoveChild, 12,
/* 45051*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45054*/   OPC_RecordNode, // #11 = $DST_SEL_W
/* 45055*/   OPC_CheckType, MVT::i32,
/* 45057*/   OPC_MoveParent,
/* 45058*/   OPC_MoveChild, 13,
/* 45060*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45063*/   OPC_RecordNode, // #12 = $RESOURCE_ID
/* 45064*/   OPC_CheckType, MVT::i32,
/* 45066*/   OPC_MoveParent,
/* 45067*/   OPC_MoveChild, 14,
/* 45069*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45072*/   OPC_RecordNode, // #13 = $SAMPLER_ID
/* 45073*/   OPC_CheckType, MVT::i32,
/* 45075*/   OPC_MoveParent,
/* 45076*/   OPC_MoveChild, 15,
/* 45078*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45081*/   OPC_RecordNode, // #14 = $COORD_TYPE_X
/* 45082*/   OPC_CheckType, MVT::i32,
/* 45084*/   OPC_MoveParent,
/* 45085*/   OPC_MoveChild, 16,
/* 45087*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45090*/   OPC_RecordNode, // #15 = $COORD_TYPE_Y
/* 45091*/   OPC_CheckType, MVT::i32,
/* 45093*/   OPC_MoveParent,
/* 45094*/   OPC_MoveChild, 17,
/* 45096*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45099*/   OPC_RecordNode, // #16 = $COORD_TYPE_Z
/* 45100*/   OPC_CheckType, MVT::i32,
/* 45102*/   OPC_MoveParent,
/* 45103*/   OPC_MoveChild, 18,
/* 45105*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45108*/   OPC_RecordNode, // #17 = $COORD_TYPE_W
/* 45109*/   OPC_CheckType, MVT::i32,
/* 45111*/   OPC_MoveParent,
/* 45112*/   OPC_CheckType, MVT::v4f32,
/* 45114*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 45116*/   OPC_EmitConvertToTarget, 1,
/* 45118*/   OPC_EmitConvertToTarget, 2,
/* 45120*/   OPC_EmitConvertToTarget, 3,
/* 45122*/   OPC_EmitConvertToTarget, 4,
/* 45124*/   OPC_EmitConvertToTarget, 5,
/* 45126*/   OPC_EmitConvertToTarget, 6,
/* 45128*/   OPC_EmitConvertToTarget, 7,
/* 45130*/   OPC_EmitConvertToTarget, 8,
/* 45132*/   OPC_EmitConvertToTarget, 9,
/* 45134*/   OPC_EmitConvertToTarget, 10,
/* 45136*/   OPC_EmitConvertToTarget, 11,
/* 45138*/   OPC_EmitConvertToTarget, 12,
/* 45140*/   OPC_EmitConvertToTarget, 13,
/* 45142*/   OPC_EmitConvertToTarget, 14,
/* 45144*/   OPC_EmitConvertToTarget, 15,
/* 45146*/   OPC_EmitConvertToTarget, 16,
/* 45148*/   OPC_EmitConvertToTarget, 17,
/* 45150*/   OPC_MorphNodeTo1, TARGET_VAL(R600::TEX_GET_GRADIENTS_H), 0,
                 MVT::v4f32, 18/*#Ops*/, 0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
             // Src: (TEXTURE_FETCH:{ *:[v4f32] } 8:{ *:[i32] }, v4f32:{ *:[v4f32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W) - Complexity = 59
             // Dst: (TEX_GET_GRADIENTS_H:{ *:[v4f32] } R600_Reg128:{ *:[v4f32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W)
/* 45174*/  /*Scope*/ 88|128,1/*216*/, /*->45392*/
/* 45176*/   OPC_CheckChild0Integer, 9, 
/* 45178*/   OPC_CheckChild0Type, MVT::i32,
/* 45180*/   OPC_RecordChild1, // #0 = $SRC_GPR
/* 45181*/   OPC_CheckChild1Type, MVT::v4f32,
/* 45183*/   OPC_RecordChild2, // #1 = $srcx
/* 45184*/   OPC_MoveChild2,
/* 45185*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45188*/   OPC_CheckType, MVT::i32,
/* 45190*/   OPC_MoveParent,
/* 45191*/   OPC_RecordChild3, // #2 = $srcy
/* 45192*/   OPC_MoveChild3,
/* 45193*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45196*/   OPC_CheckType, MVT::i32,
/* 45198*/   OPC_MoveParent,
/* 45199*/   OPC_RecordChild4, // #3 = $srcz
/* 45200*/   OPC_MoveChild4,
/* 45201*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45204*/   OPC_CheckType, MVT::i32,
/* 45206*/   OPC_MoveParent,
/* 45207*/   OPC_RecordChild5, // #4 = $srcw
/* 45208*/   OPC_MoveChild5,
/* 45209*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45212*/   OPC_CheckType, MVT::i32,
/* 45214*/   OPC_MoveParent,
/* 45215*/   OPC_RecordChild6, // #5 = $offsetx
/* 45216*/   OPC_MoveChild6,
/* 45217*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45220*/   OPC_CheckType, MVT::i32,
/* 45222*/   OPC_MoveParent,
/* 45223*/   OPC_RecordChild7, // #6 = $offsety
/* 45224*/   OPC_MoveChild7,
/* 45225*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45228*/   OPC_CheckType, MVT::i32,
/* 45230*/   OPC_MoveParent,
/* 45231*/   OPC_MoveChild, 8,
/* 45233*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45236*/   OPC_RecordNode, // #7 = $offsetz
/* 45237*/   OPC_CheckType, MVT::i32,
/* 45239*/   OPC_MoveParent,
/* 45240*/   OPC_MoveChild, 9,
/* 45242*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45245*/   OPC_RecordNode, // #8 = $DST_SEL_X
/* 45246*/   OPC_CheckType, MVT::i32,
/* 45248*/   OPC_MoveParent,
/* 45249*/   OPC_MoveChild, 10,
/* 45251*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45254*/   OPC_RecordNode, // #9 = $DST_SEL_Y
/* 45255*/   OPC_CheckType, MVT::i32,
/* 45257*/   OPC_MoveParent,
/* 45258*/   OPC_MoveChild, 11,
/* 45260*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45263*/   OPC_RecordNode, // #10 = $DST_SEL_Z
/* 45264*/   OPC_CheckType, MVT::i32,
/* 45266*/   OPC_MoveParent,
/* 45267*/   OPC_MoveChild, 12,
/* 45269*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45272*/   OPC_RecordNode, // #11 = $DST_SEL_W
/* 45273*/   OPC_CheckType, MVT::i32,
/* 45275*/   OPC_MoveParent,
/* 45276*/   OPC_MoveChild, 13,
/* 45278*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45281*/   OPC_RecordNode, // #12 = $RESOURCE_ID
/* 45282*/   OPC_CheckType, MVT::i32,
/* 45284*/   OPC_MoveParent,
/* 45285*/   OPC_MoveChild, 14,
/* 45287*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45290*/   OPC_RecordNode, // #13 = $SAMPLER_ID
/* 45291*/   OPC_CheckType, MVT::i32,
/* 45293*/   OPC_MoveParent,
/* 45294*/   OPC_MoveChild, 15,
/* 45296*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45299*/   OPC_RecordNode, // #14 = $COORD_TYPE_X
/* 45300*/   OPC_CheckType, MVT::i32,
/* 45302*/   OPC_MoveParent,
/* 45303*/   OPC_MoveChild, 16,
/* 45305*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45308*/   OPC_RecordNode, // #15 = $COORD_TYPE_Y
/* 45309*/   OPC_CheckType, MVT::i32,
/* 45311*/   OPC_MoveParent,
/* 45312*/   OPC_MoveChild, 17,
/* 45314*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45317*/   OPC_RecordNode, // #16 = $COORD_TYPE_Z
/* 45318*/   OPC_CheckType, MVT::i32,
/* 45320*/   OPC_MoveParent,
/* 45321*/   OPC_MoveChild, 18,
/* 45323*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45326*/   OPC_RecordNode, // #17 = $COORD_TYPE_W
/* 45327*/   OPC_CheckType, MVT::i32,
/* 45329*/   OPC_MoveParent,
/* 45330*/   OPC_CheckType, MVT::v4f32,
/* 45332*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 45334*/   OPC_EmitConvertToTarget, 1,
/* 45336*/   OPC_EmitConvertToTarget, 2,
/* 45338*/   OPC_EmitConvertToTarget, 3,
/* 45340*/   OPC_EmitConvertToTarget, 4,
/* 45342*/   OPC_EmitConvertToTarget, 5,
/* 45344*/   OPC_EmitConvertToTarget, 6,
/* 45346*/   OPC_EmitConvertToTarget, 7,
/* 45348*/   OPC_EmitConvertToTarget, 8,
/* 45350*/   OPC_EmitConvertToTarget, 9,
/* 45352*/   OPC_EmitConvertToTarget, 10,
/* 45354*/   OPC_EmitConvertToTarget, 11,
/* 45356*/   OPC_EmitConvertToTarget, 12,
/* 45358*/   OPC_EmitConvertToTarget, 13,
/* 45360*/   OPC_EmitConvertToTarget, 14,
/* 45362*/   OPC_EmitConvertToTarget, 15,
/* 45364*/   OPC_EmitConvertToTarget, 16,
/* 45366*/   OPC_EmitConvertToTarget, 17,
/* 45368*/   OPC_MorphNodeTo1, TARGET_VAL(R600::TEX_GET_GRADIENTS_V), 0,
                 MVT::v4f32, 18/*#Ops*/, 0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
             // Src: (TEXTURE_FETCH:{ *:[v4f32] } 9:{ *:[i32] }, v4f32:{ *:[v4f32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W) - Complexity = 59
             // Dst: (TEX_GET_GRADIENTS_V:{ *:[v4f32] } R600_Reg128:{ *:[v4f32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W)
/* 45392*/  /*Scope*/ 88|128,1/*216*/, /*->45610*/
/* 45394*/   OPC_CheckChild0Integer, 10, 
/* 45396*/   OPC_CheckChild0Type, MVT::i32,
/* 45398*/   OPC_RecordChild1, // #0 = $SRC_GPR
/* 45399*/   OPC_CheckChild1Type, MVT::v4i32,
/* 45401*/   OPC_RecordChild2, // #1 = $srcx
/* 45402*/   OPC_MoveChild2,
/* 45403*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45406*/   OPC_CheckType, MVT::i32,
/* 45408*/   OPC_MoveParent,
/* 45409*/   OPC_RecordChild3, // #2 = $srcy
/* 45410*/   OPC_MoveChild3,
/* 45411*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45414*/   OPC_CheckType, MVT::i32,
/* 45416*/   OPC_MoveParent,
/* 45417*/   OPC_RecordChild4, // #3 = $srcz
/* 45418*/   OPC_MoveChild4,
/* 45419*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45422*/   OPC_CheckType, MVT::i32,
/* 45424*/   OPC_MoveParent,
/* 45425*/   OPC_RecordChild5, // #4 = $srcw
/* 45426*/   OPC_MoveChild5,
/* 45427*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45430*/   OPC_CheckType, MVT::i32,
/* 45432*/   OPC_MoveParent,
/* 45433*/   OPC_RecordChild6, // #5 = $offsetx
/* 45434*/   OPC_MoveChild6,
/* 45435*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45438*/   OPC_CheckType, MVT::i32,
/* 45440*/   OPC_MoveParent,
/* 45441*/   OPC_RecordChild7, // #6 = $offsety
/* 45442*/   OPC_MoveChild7,
/* 45443*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45446*/   OPC_CheckType, MVT::i32,
/* 45448*/   OPC_MoveParent,
/* 45449*/   OPC_MoveChild, 8,
/* 45451*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45454*/   OPC_RecordNode, // #7 = $offsetz
/* 45455*/   OPC_CheckType, MVT::i32,
/* 45457*/   OPC_MoveParent,
/* 45458*/   OPC_MoveChild, 9,
/* 45460*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45463*/   OPC_RecordNode, // #8 = $DST_SEL_X
/* 45464*/   OPC_CheckType, MVT::i32,
/* 45466*/   OPC_MoveParent,
/* 45467*/   OPC_MoveChild, 10,
/* 45469*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45472*/   OPC_RecordNode, // #9 = $DST_SEL_Y
/* 45473*/   OPC_CheckType, MVT::i32,
/* 45475*/   OPC_MoveParent,
/* 45476*/   OPC_MoveChild, 11,
/* 45478*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45481*/   OPC_RecordNode, // #10 = $DST_SEL_Z
/* 45482*/   OPC_CheckType, MVT::i32,
/* 45484*/   OPC_MoveParent,
/* 45485*/   OPC_MoveChild, 12,
/* 45487*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45490*/   OPC_RecordNode, // #11 = $DST_SEL_W
/* 45491*/   OPC_CheckType, MVT::i32,
/* 45493*/   OPC_MoveParent,
/* 45494*/   OPC_MoveChild, 13,
/* 45496*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45499*/   OPC_RecordNode, // #12 = $RESOURCE_ID
/* 45500*/   OPC_CheckType, MVT::i32,
/* 45502*/   OPC_MoveParent,
/* 45503*/   OPC_MoveChild, 14,
/* 45505*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45508*/   OPC_RecordNode, // #13 = $SAMPLER_ID
/* 45509*/   OPC_CheckType, MVT::i32,
/* 45511*/   OPC_MoveParent,
/* 45512*/   OPC_MoveChild, 15,
/* 45514*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45517*/   OPC_RecordNode, // #14 = $COORD_TYPE_X
/* 45518*/   OPC_CheckType, MVT::i32,
/* 45520*/   OPC_MoveParent,
/* 45521*/   OPC_MoveChild, 16,
/* 45523*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45526*/   OPC_RecordNode, // #15 = $COORD_TYPE_Y
/* 45527*/   OPC_CheckType, MVT::i32,
/* 45529*/   OPC_MoveParent,
/* 45530*/   OPC_MoveChild, 17,
/* 45532*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45535*/   OPC_RecordNode, // #16 = $COORD_TYPE_Z
/* 45536*/   OPC_CheckType, MVT::i32,
/* 45538*/   OPC_MoveParent,
/* 45539*/   OPC_MoveChild, 18,
/* 45541*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/* 45544*/   OPC_RecordNode, // #17 = $COORD_TYPE_W
/* 45545*/   OPC_CheckType, MVT::i32,
/* 45547*/   OPC_MoveParent,
/* 45548*/   OPC_CheckType, MVT::v4f32,
/* 45550*/   OPC_CheckPatternPredicate, 2, // (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
/* 45552*/   OPC_EmitConvertToTarget, 1,
/* 45554*/   OPC_EmitConvertToTarget, 2,
/* 45556*/   OPC_EmitConvertToTarget, 3,
/* 45558*/   OPC_EmitConvertToTarget, 4,
/* 45560*/   OPC_EmitConvertToTarget, 5,
/* 45562*/   OPC_EmitConvertToTarget, 6,
/* 45564*/   OPC_EmitConvertToTarget, 7,
/* 45566*/   OPC_EmitConvertToTarget, 8,
/* 45568*/   OPC_EmitConvertToTarget, 9,
/* 45570*/   OPC_EmitConvertToTarget, 10,
/* 45572*/   OPC_EmitConvertToTarget, 11,
/* 45574*/   OPC_EmitConvertToTarget, 12,
/* 45576*/   OPC_EmitConvertToTarget, 13,
/* 45578*/   OPC_EmitConvertToTarget, 14,
/* 45580*/   OPC_EmitConvertToTarget, 15,
/* 45582*/   OPC_EmitConvertToTarget, 16,
/* 45584*/   OPC_EmitConvertToTarget, 17,
/* 45586*/   OPC_MorphNodeTo1, TARGET_VAL(R600::TEX_LDPTR), 0,
                 MVT::v4f32, 18/*#Ops*/, 0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
             // Src: (TEXTURE_FETCH:{ *:[v4f32] } 10:{ *:[i32] }, v4i32:{ *:[v4i32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W) - Complexity = 59
             // Dst: (TEX_LDPTR:{ *:[v4f32] } R600_Reg128:{ *:[v4i32] }:$SRC_GPR, (imm:{ *:[i32] }):$srcx, (imm:{ *:[i32] }):$srcy, (imm:{ *:[i32] }):$srcz, (imm:{ *:[i32] }):$srcw, (imm:{ *:[i32] }):$offsetx, (imm:{ *:[i32] }):$offsety, (imm:{ *:[i32] }):$offsetz, (imm:{ *:[i32] }):$DST_SEL_X, (imm:{ *:[i32] }):$DST_SEL_Y, (imm:{ *:[i32] }):$DST_SEL_Z, (imm:{ *:[i32] }):$DST_SEL_W, (imm:{ *:[i32] }):$RESOURCE_ID, (imm:{ *:[i32] }):$SAMPLER_ID, (imm:{ *:[i32] }):$COORD_TYPE_X, (imm:{ *:[i32] }):$COORD_TYPE_Y, (imm:{ *:[i32] }):$COORD_TYPE_Z, (imm:{ *:[i32] }):$COORD_TYPE_W)
/* 45610*/  0, /*End of Scope*/
/* 45611*/ 0, // EndSwitchOpcode
    0
  }; // Total Array size is 45613 bytes

  // Opcode Histogram:
  // #OPC_Scope                                = 147
  // #OPC_RecordNode                           = 145
  // #OPC_RecordChild                          = 422
  // #OPC_RecordMemRef                         = 14
  // #OPC_CaptureGlueInput                     = 1
  // #OPC_MoveChild                            = 288
  // #OPC_MoveParent                           = 396
  // #OPC_CheckSame                            = 0
  // #OPC_CheckChildSame                       = 90
  // #OPC_CheckPatternPredicate                = 402
  // #OPC_CheckPredicate                       = 115
  // #OPC_CheckOpcode                          = 277
  // #OPC_SwitchOpcode                         = 5
  // #OPC_CheckType                            = 434
  // #OPC_SwitchType                           = 16
  // #OPC_CheckChildType                       = 105
  // #OPC_CheckInteger                         = 0
  // #OPC_CheckChildInteger                    = 92
  // #OPC_CheckCondCode                        = 42
  // #OPC_CheckChild2CondCode                  = 0
  // #OPC_CheckValueType                       = 3
  // #OPC_CheckComplexPat                      = 36
  // #OPC_CheckAndImm                          = 0
  // #OPC_CheckOrImm                           = 0
  // #OPC_CheckImmAllOnesV                     = 0
  // #OPC_CheckImmAllZerosV                    = 0
  // #OPC_CheckFoldableChainNode               = 0
  // #OPC_EmitInteger                          = 5327
  // #OPC_EmitStringInteger                    = 383
  // #OPC_EmitRegister                         = 385
  // #OPC_EmitConvertToTarget                  = 220
  // #OPC_EmitMergeInputChains                 = 94
  // #OPC_EmitCopyToReg                        = 0
  // #OPC_EmitNode                             = 428
  // #OPC_EmitNodeXForm                        = 1
  // #OPC_CompleteMatch                        = 7
  // #OPC_MorphNodeTo                          = 408

  #undef TARGET_VAL
  SelectCodeCommon(N, MatcherTable,sizeof(MatcherTable));
}
#endif // GET_DAGISEL_BODY

#ifdef GET_DAGISEL_DECL
bool CheckPatternPredicate(unsigned PredNo) const override;
#endif
#if defined(GET_DAGISEL_BODY) || DAGISEL_INLINE
bool DAGISEL_CLASS_COLONCOLON CheckPatternPredicate(unsigned PredNo) const
#if DAGISEL_INLINE
  override
#endif
{
  switch (PredNo) {
  default: llvm_unreachable("Invalid predicate in table?");
  case 0: return (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS);
  case 1: return (Subtarget->getGeneration() <= AMDGPUSubtarget::R700);
  case 2: return (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS);
  case 3: return (Subtarget->hasCaymanISA());
  case 4: return (Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && !Subtarget->hasCaymanISA());
  case 5: return (Subtarget->getGeneration() == AMDGPUSubtarget::R700);
  case 6: return (Subtarget->hasFMA()) && (Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||Subtarget->getGeneration() == AMDGPUSubtarget::NORTHERN_ISLANDS);
  }
}
#endif // GET_DAGISEL_BODY

#ifdef GET_DAGISEL_DECL
bool CheckNodePredicate(SDNode *Node, unsigned PredNo) const override;
#endif
#if defined(GET_DAGISEL_BODY) || DAGISEL_INLINE
bool DAGISEL_CLASS_COLONCOLON CheckNodePredicate(SDNode *Node, unsigned PredNo) const
#if DAGISEL_INLINE
  override
#endif
{
  switch (PredNo) {
  default: llvm_unreachable("Invalid predicate in table?");
  case 0: { 
    // Predicate_shl_oneuse
    // Predicate_add_oneuse
    // Predicate_xor_oneuse
    // Predicate_srl_oneuse
    SDNode *N = Node;
    (void)N;
 return N->hasOneUse(); 
  }
  case 1: { 
    // Predicate_IMMZeroBasedBitfieldMask
    int64_t Imm = cast<ConstantSDNode>(Node)->getSExtValue();

  return isMask_32(Imm);

  }
  case 2: { 
    // Predicate_FP_ONE
    auto *N = cast<ConstantFPSDNode>(Node);
    (void)N;
return N->isExactlyValue(1.0);
  }
  case 3: { 
    // Predicate_FP_ZERO
    auto *N = cast<ConstantFPSDNode>(Node);
    (void)N;
return N->getValueAPF().isZero();
  }
  case 4: { 
    // Predicate_unindexedload
    SDNode *N = Node;
    (void)N;
if (cast<LoadSDNode>(N)->getAddressingMode() != ISD::UNINDEXED) return false;
return true;

  }
  case 5: { 
    // Predicate_load_private
    SDNode *N = Node;
    (void)N;
unsigned AddrSpace = cast<MemSDNode>(N)->getAddressSpace();
 if (AddrSpace != 5)
return false;
if (cast<LoadSDNode>(N)->getExtensionType() != ISD::NON_EXTLOAD) return false;
return true;

  }
  case 6: { 
    // Predicate_az_extload
    SDNode *N = Node;
    (void)N;

  LoadSDNode *L = cast<LoadSDNode>(N);
  return L->getExtensionType() == ISD::ZEXTLOAD ||
         L->getExtensionType() == ISD::EXTLOAD;

  }
  case 7: { 
    // Predicate_az_extloadi8
    SDNode *N = Node;
    (void)N;

  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i8;

  }
  case 8: { 
    // Predicate_vtx_id3_az_extloadi8
    // Predicate_vtx_id3_az_extloadi16
    // Predicate_vtx_id3_load
    SDNode *N = Node;
    (void)N;
 return isConstantLoad(cast<LoadSDNode>(N), 0) ||
            (cast<LoadSDNode>(N)->getAddressSpace() == AMDGPUAS::PARAM_I_ADDRESS); 
  }
  case 9: { 
    // Predicate_az_extloadi16
    SDNode *N = Node;
    (void)N;

  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i16;

  }
  case 10: { 
    // Predicate_load
    SDNode *N = Node;
    (void)N;
if (cast<LoadSDNode>(N)->getExtensionType() != ISD::NON_EXTLOAD) return false;
return true;

  }
  case 11: { 
    // Predicate_vtx_id2_az_extloadi8
    // Predicate_vtx_id2_az_extloadi16
    // Predicate_vtx_id2_load
    SDNode *N = Node;
    (void)N;

  const MemSDNode *LD = cast<MemSDNode>(N);
  return LD->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS &&
         isa<GlobalValue>(getUnderlyingObject(
         LD->getMemOperand()->getValue()));

  }
  case 12: { 
    // Predicate_vtx_id1_az_extloadi8
    // Predicate_vtx_id1_az_extloadi16
    // Predicate_vtx_id1_load
    SDNode *N = Node;
    (void)N;

  const MemSDNode *LD = cast<MemSDNode>(N);
  return LD->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS ||
         (LD->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS &&
           !isa<GlobalValue>(getUnderlyingObject(
           LD->getMemOperand()->getValue())));

  }
  case 13: { 
    // Predicate_load_local
    SDNode *N = Node;
    (void)N;
unsigned AddrSpace = cast<MemSDNode>(N)->getAddressSpace();
 if (AddrSpace != 3)
return false;
if (cast<LoadSDNode>(N)->getExtensionType() != ISD::NON_EXTLOAD) return false;
return true;

  }
  case 14: { 
    // Predicate_sextload
    SDNode *N = Node;
    (void)N;
if (cast<LoadSDNode>(N)->getExtensionType() != ISD::SEXTLOAD) return false;
return true;

  }
  case 15: { 
    // Predicate_sextloadi8_local
    // Predicate_truncstorei8_local
    SDNode *N = Node;
    (void)N;
unsigned AddrSpace = cast<MemSDNode>(N)->getAddressSpace();
 if (AddrSpace != 3)
return false;
if (cast<MemSDNode>(N)->getMemoryVT() != MVT::i8) return false;
return true;

  }
  case 16: { 
    // Predicate_sextloadi16_local
    // Predicate_truncstorei16_local
    SDNode *N = Node;
    (void)N;
unsigned AddrSpace = cast<MemSDNode>(N)->getAddressSpace();
 if (AddrSpace != 3)
return false;
if (cast<MemSDNode>(N)->getMemoryVT() != MVT::i16) return false;
return true;

  }
  case 17: { 
    // Predicate_unindexedstore
    SDNode *N = Node;
    (void)N;
if (cast<StoreSDNode>(N)->getAddressingMode() != ISD::UNINDEXED) return false;
return true;

  }
  case 18: { 
    // Predicate_store_private
    SDNode *N = Node;
    (void)N;
unsigned AddrSpace = cast<MemSDNode>(N)->getAddressSpace();
 if (AddrSpace != 5)
return false;
 if (cast<StoreSDNode>(N)->isTruncatingStore()) return false;
return true;

  }
  case 19: { 
    // Predicate_store_local
    SDNode *N = Node;
    (void)N;
unsigned AddrSpace = cast<MemSDNode>(N)->getAddressSpace();
 if (AddrSpace != 3)
return false;
 if (cast<StoreSDNode>(N)->isTruncatingStore()) return false;
return true;

  }
  case 20: { 
    // Predicate_truncstore
    SDNode *N = Node;
    (void)N;
 if (!cast<StoreSDNode>(N)->isTruncatingStore()) return false;
return true;

  }
  case 21: { 
    // Predicate_store_global
    SDNode *N = Node;
    (void)N;
unsigned AddrSpace = cast<MemSDNode>(N)->getAddressSpace();
 if (AddrSpace != 1)
return false;
 if (cast<StoreSDNode>(N)->isTruncatingStore()) return false;
return true;

  }
  case 22: { 
    // Predicate_atomic_load_add_global_noret_32
    // Predicate_atomic_load_sub_global_noret_32
    // Predicate_atomic_swap_global_noret_32
    // Predicate_atomic_load_min_global_noret_32
    // Predicate_atomic_load_umin_global_noret_32
    // Predicate_atomic_load_max_global_noret_32
    // Predicate_atomic_load_umax_global_noret_32
    // Predicate_atomic_load_and_global_noret_32
    // Predicate_atomic_load_or_global_noret_32
    // Predicate_atomic_load_xor_global_noret_32
    SDNode *N = Node;
    (void)N;
unsigned AddrSpace = cast<MemSDNode>(N)->getAddressSpace();
 if (AddrSpace != 1 && AddrSpace != 4)
return false;
if (cast<MemSDNode>(N)->getMemoryVT() != MVT::i32) return false;
return (SDValue(N, 0).use_empty());
  }
  case 23: { 
    // Predicate_atomic_load_add_local_32
    // Predicate_atomic_load_sub_local_32
    // Predicate_atomic_swap_local_32
    // Predicate_atomic_load_min_local_32
    // Predicate_atomic_load_umin_local_32
    // Predicate_atomic_load_max_local_32
    // Predicate_atomic_load_umax_local_32
    // Predicate_atomic_load_and_local_32
    // Predicate_atomic_load_or_local_32
    // Predicate_atomic_load_xor_local_32
    // Predicate_atomic_cmp_swap_local_32
    SDNode *N = Node;
    (void)N;
unsigned AddrSpace = cast<MemSDNode>(N)->getAddressSpace();
 if (AddrSpace != 3)
return false;
if (cast<MemSDNode>(N)->getMemoryVT() != MVT::i32) return false;
return true;

  }
  case 24: { 
    // Predicate_mskor_global
    SDNode *N = Node;
    (void)N;

  return cast<MemSDNode>(N)->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS;

  }
  case 25: { 
    // Predicate_atomic_cmp_swap_global_noret
    SDNode *N = Node;
    (void)N;
return cast<MemSDNode>(N)->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS && (SDValue(N, 0).use_empty());
  }
  }
}
#endif // GET_DAGISEL_BODY

#ifdef GET_DAGISEL_DECL
bool CheckComplexPattern(SDNode *Root, SDNode *Parent,
      SDValue N, unsigned PatternNo,
      SmallVectorImpl<std::pair<SDValue, SDNode*>> &Result) override;
#endif
#if defined(GET_DAGISEL_BODY) || DAGISEL_INLINE
bool DAGISEL_CLASS_COLONCOLON CheckComplexPattern(SDNode *Root, SDNode *Parent,
      SDValue N, unsigned PatternNo,
      SmallVectorImpl<std::pair<SDValue, SDNode*>> &Result)
#if DAGISEL_INLINE
  override
#endif
{
  unsigned NextRes = Result.size();
  switch (PatternNo) {
  default: llvm_unreachable("Invalid pattern # in table?");
  case 0:
    Result.resize(NextRes+2);
  return SelectADDRIndirect(N, Result[NextRes+0].first, Result[NextRes+1].first);
  case 1:
    Result.resize(NextRes+2);
  return SelectADDRVTX_READ(N, Result[NextRes+0].first, Result[NextRes+1].first);
  case 2:
    Result.resize(NextRes+1);
  return SelectGlobalValueConstantOffset(N, Result[NextRes+0].first);
  case 3:
    Result.resize(NextRes+2);
  return SelectGlobalValueVariableOffset(N, Result[NextRes+0].first, Result[NextRes+1].first);
  }
}
#endif // GET_DAGISEL_BODY

#ifdef GET_DAGISEL_DECL
SDValue RunSDNodeXForm(SDValue V, unsigned XFormNo) override;
#endif
#if defined(GET_DAGISEL_BODY) || DAGISEL_INLINE
SDValue DAGISEL_CLASS_COLONCOLON RunSDNodeXForm(SDValue V, unsigned XFormNo)
#if DAGISEL_INLINE
  override
#endif
{
  switch (XFormNo) {
  default: llvm_unreachable("Invalid xform # in table?");
  case 0: {  // IMMPopCount
    ConstantSDNode *N = cast<ConstantSDNode>(V.getNode());

  return CurDAG->getTargetConstant(countPopulation(N->getZExtValue()), SDLoc(N),
                                   MVT::i32);

  }
  }
}
#endif // GET_DAGISEL_BODY


#ifdef DAGISEL_INLINE
#undef DAGISEL_INLINE
#endif
#ifdef DAGISEL_CLASS_COLONCOLON
#undef DAGISEL_CLASS_COLONCOLON
#endif
#ifdef GET_DAGISEL_DECL
#undef GET_DAGISEL_DECL
#endif
#ifdef GET_DAGISEL_BODY
#undef GET_DAGISEL_BODY
#endif
