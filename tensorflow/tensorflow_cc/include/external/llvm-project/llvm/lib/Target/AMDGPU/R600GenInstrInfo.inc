/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Target Instruction Enum Values and Descriptors                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_INSTRINFO_ENUM
#undef GET_INSTRINFO_ENUM
namespace llvm {

namespace R600 {
  enum {
    PHI	= 0,
    INLINEASM	= 1,
    INLINEASM_BR	= 2,
    CFI_INSTRUCTION	= 3,
    EH_LABEL	= 4,
    GC_LABEL	= 5,
    ANNOTATION_LABEL	= 6,
    KILL	= 7,
    EXTRACT_SUBREG	= 8,
    INSERT_SUBREG	= 9,
    IMPLICIT_DEF	= 10,
    SUBREG_TO_REG	= 11,
    COPY_TO_REGCLASS	= 12,
    DBG_VALUE	= 13,
    DBG_INSTR_REF	= 14,
    DBG_LABEL	= 15,
    REG_SEQUENCE	= 16,
    COPY	= 17,
    BUNDLE	= 18,
    LIFETIME_START	= 19,
    LIFETIME_END	= 20,
    STACKMAP	= 21,
    FENTRY_CALL	= 22,
    PATCHPOINT	= 23,
    LOAD_STACK_GUARD	= 24,
    PREALLOCATED_SETUP	= 25,
    PREALLOCATED_ARG	= 26,
    STATEPOINT	= 27,
    LOCAL_ESCAPE	= 28,
    FAULTING_OP	= 29,
    PATCHABLE_OP	= 30,
    PATCHABLE_FUNCTION_ENTER	= 31,
    PATCHABLE_RET	= 32,
    PATCHABLE_FUNCTION_EXIT	= 33,
    PATCHABLE_TAIL_CALL	= 34,
    PATCHABLE_EVENT_CALL	= 35,
    PATCHABLE_TYPED_EVENT_CALL	= 36,
    ICALL_BRANCH_FUNNEL	= 37,
    G_ADD	= 38,
    G_SUB	= 39,
    G_MUL	= 40,
    G_SDIV	= 41,
    G_UDIV	= 42,
    G_SREM	= 43,
    G_UREM	= 44,
    G_AND	= 45,
    G_OR	= 46,
    G_XOR	= 47,
    G_IMPLICIT_DEF	= 48,
    G_PHI	= 49,
    G_FRAME_INDEX	= 50,
    G_GLOBAL_VALUE	= 51,
    G_EXTRACT	= 52,
    G_UNMERGE_VALUES	= 53,
    G_INSERT	= 54,
    G_MERGE_VALUES	= 55,
    G_BUILD_VECTOR	= 56,
    G_BUILD_VECTOR_TRUNC	= 57,
    G_CONCAT_VECTORS	= 58,
    G_PTRTOINT	= 59,
    G_INTTOPTR	= 60,
    G_BITCAST	= 61,
    G_FREEZE	= 62,
    G_INTRINSIC_TRUNC	= 63,
    G_INTRINSIC_ROUND	= 64,
    G_INTRINSIC_LRINT	= 65,
    G_INTRINSIC_ROUNDEVEN	= 66,
    G_READCYCLECOUNTER	= 67,
    G_LOAD	= 68,
    G_SEXTLOAD	= 69,
    G_ZEXTLOAD	= 70,
    G_INDEXED_LOAD	= 71,
    G_INDEXED_SEXTLOAD	= 72,
    G_INDEXED_ZEXTLOAD	= 73,
    G_STORE	= 74,
    G_INDEXED_STORE	= 75,
    G_ATOMIC_CMPXCHG_WITH_SUCCESS	= 76,
    G_ATOMIC_CMPXCHG	= 77,
    G_ATOMICRMW_XCHG	= 78,
    G_ATOMICRMW_ADD	= 79,
    G_ATOMICRMW_SUB	= 80,
    G_ATOMICRMW_AND	= 81,
    G_ATOMICRMW_NAND	= 82,
    G_ATOMICRMW_OR	= 83,
    G_ATOMICRMW_XOR	= 84,
    G_ATOMICRMW_MAX	= 85,
    G_ATOMICRMW_MIN	= 86,
    G_ATOMICRMW_UMAX	= 87,
    G_ATOMICRMW_UMIN	= 88,
    G_ATOMICRMW_FADD	= 89,
    G_ATOMICRMW_FSUB	= 90,
    G_FENCE	= 91,
    G_BRCOND	= 92,
    G_BRINDIRECT	= 93,
    G_INTRINSIC	= 94,
    G_INTRINSIC_W_SIDE_EFFECTS	= 95,
    G_ANYEXT	= 96,
    G_TRUNC	= 97,
    G_CONSTANT	= 98,
    G_FCONSTANT	= 99,
    G_VASTART	= 100,
    G_VAARG	= 101,
    G_SEXT	= 102,
    G_SEXT_INREG	= 103,
    G_ZEXT	= 104,
    G_SHL	= 105,
    G_LSHR	= 106,
    G_ASHR	= 107,
    G_FSHL	= 108,
    G_FSHR	= 109,
    G_ICMP	= 110,
    G_FCMP	= 111,
    G_SELECT	= 112,
    G_UADDO	= 113,
    G_UADDE	= 114,
    G_USUBO	= 115,
    G_USUBE	= 116,
    G_SADDO	= 117,
    G_SADDE	= 118,
    G_SSUBO	= 119,
    G_SSUBE	= 120,
    G_UMULO	= 121,
    G_SMULO	= 122,
    G_UMULH	= 123,
    G_SMULH	= 124,
    G_UADDSAT	= 125,
    G_SADDSAT	= 126,
    G_USUBSAT	= 127,
    G_SSUBSAT	= 128,
    G_USHLSAT	= 129,
    G_SSHLSAT	= 130,
    G_SMULFIX	= 131,
    G_UMULFIX	= 132,
    G_SMULFIXSAT	= 133,
    G_UMULFIXSAT	= 134,
    G_SDIVFIX	= 135,
    G_UDIVFIX	= 136,
    G_SDIVFIXSAT	= 137,
    G_UDIVFIXSAT	= 138,
    G_FADD	= 139,
    G_FSUB	= 140,
    G_FMUL	= 141,
    G_FMA	= 142,
    G_FMAD	= 143,
    G_FDIV	= 144,
    G_FREM	= 145,
    G_FPOW	= 146,
    G_FPOWI	= 147,
    G_FEXP	= 148,
    G_FEXP2	= 149,
    G_FLOG	= 150,
    G_FLOG2	= 151,
    G_FLOG10	= 152,
    G_FNEG	= 153,
    G_FPEXT	= 154,
    G_FPTRUNC	= 155,
    G_FPTOSI	= 156,
    G_FPTOUI	= 157,
    G_SITOFP	= 158,
    G_UITOFP	= 159,
    G_FABS	= 160,
    G_FCOPYSIGN	= 161,
    G_FCANONICALIZE	= 162,
    G_FMINNUM	= 163,
    G_FMAXNUM	= 164,
    G_FMINNUM_IEEE	= 165,
    G_FMAXNUM_IEEE	= 166,
    G_FMINIMUM	= 167,
    G_FMAXIMUM	= 168,
    G_PTR_ADD	= 169,
    G_PTRMASK	= 170,
    G_SMIN	= 171,
    G_SMAX	= 172,
    G_UMIN	= 173,
    G_UMAX	= 174,
    G_ABS	= 175,
    G_BR	= 176,
    G_BRJT	= 177,
    G_INSERT_VECTOR_ELT	= 178,
    G_EXTRACT_VECTOR_ELT	= 179,
    G_SHUFFLE_VECTOR	= 180,
    G_CTTZ	= 181,
    G_CTTZ_ZERO_UNDEF	= 182,
    G_CTLZ	= 183,
    G_CTLZ_ZERO_UNDEF	= 184,
    G_CTPOP	= 185,
    G_BSWAP	= 186,
    G_BITREVERSE	= 187,
    G_FCEIL	= 188,
    G_FCOS	= 189,
    G_FSIN	= 190,
    G_FSQRT	= 191,
    G_FFLOOR	= 192,
    G_FRINT	= 193,
    G_FNEARBYINT	= 194,
    G_ADDRSPACE_CAST	= 195,
    G_BLOCK_ADDR	= 196,
    G_JUMP_TABLE	= 197,
    G_DYN_STACKALLOC	= 198,
    G_STRICT_FADD	= 199,
    G_STRICT_FSUB	= 200,
    G_STRICT_FMUL	= 201,
    G_STRICT_FDIV	= 202,
    G_STRICT_FREM	= 203,
    G_STRICT_FMA	= 204,
    G_STRICT_FSQRT	= 205,
    G_READ_REGISTER	= 206,
    G_WRITE_REGISTER	= 207,
    G_MEMCPY	= 208,
    G_MEMMOVE	= 209,
    G_MEMSET	= 210,
    G_VECREDUCE_SEQ_FADD	= 211,
    G_VECREDUCE_SEQ_FMUL	= 212,
    G_VECREDUCE_FADD	= 213,
    G_VECREDUCE_FMUL	= 214,
    G_VECREDUCE_FMAX	= 215,
    G_VECREDUCE_FMIN	= 216,
    G_VECREDUCE_ADD	= 217,
    G_VECREDUCE_MUL	= 218,
    G_VECREDUCE_AND	= 219,
    G_VECREDUCE_OR	= 220,
    G_VECREDUCE_XOR	= 221,
    G_VECREDUCE_SMAX	= 222,
    G_VECREDUCE_SMIN	= 223,
    G_VECREDUCE_UMAX	= 224,
    G_VECREDUCE_UMIN	= 225,
    BRANCH	= 226,
    BRANCH_COND_f32	= 227,
    BRANCH_COND_i32	= 228,
    BREAK	= 229,
    BREAKC_f32	= 230,
    BREAKC_i32	= 231,
    BREAK_LOGICALNZ_f32	= 232,
    BREAK_LOGICALNZ_i32	= 233,
    BREAK_LOGICALZ_f32	= 234,
    BREAK_LOGICALZ_i32	= 235,
    CONST_COPY	= 236,
    CONTINUE	= 237,
    CONTINUEC_f32	= 238,
    CONTINUEC_i32	= 239,
    CONTINUE_LOGICALNZ_f32	= 240,
    CONTINUE_LOGICALNZ_i32	= 241,
    CONTINUE_LOGICALZ_f32	= 242,
    CONTINUE_LOGICALZ_i32	= 243,
    CUBE_eg_pseudo	= 244,
    CUBE_r600_pseudo	= 245,
    DEFAULT	= 246,
    DOT_4	= 247,
    DUMMY_CHAIN	= 248,
    ELSE	= 249,
    END	= 250,
    ENDFUNC	= 251,
    ENDIF	= 252,
    ENDLOOP	= 253,
    ENDMAIN	= 254,
    ENDSWITCH	= 255,
    FABS_R600	= 256,
    FNEG_R600	= 257,
    FUNC	= 258,
    IFC_f32	= 259,
    IFC_i32	= 260,
    IF_LOGICALNZ_f32	= 261,
    IF_LOGICALNZ_i32	= 262,
    IF_LOGICALZ_f32	= 263,
    IF_LOGICALZ_i32	= 264,
    IF_PREDICATE_SET	= 265,
    JUMP	= 266,
    JUMP_COND	= 267,
    MASK_WRITE	= 268,
    MOV_IMM_F32	= 269,
    MOV_IMM_GLOBAL_ADDR	= 270,
    MOV_IMM_I32	= 271,
    PRED_X	= 272,
    R600_EXTRACT_ELT_V2	= 273,
    R600_EXTRACT_ELT_V4	= 274,
    R600_INSERT_ELT_V2	= 275,
    R600_INSERT_ELT_V4	= 276,
    R600_RegisterLoad	= 277,
    R600_RegisterStore	= 278,
    RETDYN	= 279,
    RETURN	= 280,
    TXD	= 281,
    TXD_SHADOW	= 282,
    WHILELOOP	= 283,
    ADD	= 284,
    ADDC_UINT	= 285,
    ADD_INT	= 286,
    ALU_CLAUSE	= 287,
    AND_INT	= 288,
    ASHR_eg	= 289,
    ASHR_r600	= 290,
    BCNT_INT	= 291,
    BFE_INT_eg	= 292,
    BFE_UINT_eg	= 293,
    BFI_INT_eg	= 294,
    BFM_INT_eg	= 295,
    BIT_ALIGN_INT_eg	= 296,
    CEIL	= 297,
    CF_ALU	= 298,
    CF_ALU_BREAK	= 299,
    CF_ALU_CONTINUE	= 300,
    CF_ALU_ELSE_AFTER	= 301,
    CF_ALU_POP_AFTER	= 302,
    CF_ALU_PUSH_BEFORE	= 303,
    CF_CALL_FS_EG	= 304,
    CF_CALL_FS_R600	= 305,
    CF_CONTINUE_EG	= 306,
    CF_CONTINUE_R600	= 307,
    CF_ELSE_EG	= 308,
    CF_ELSE_R600	= 309,
    CF_END_CM	= 310,
    CF_END_EG	= 311,
    CF_END_R600	= 312,
    CF_JUMP_EG	= 313,
    CF_JUMP_R600	= 314,
    CF_PUSH_EG	= 315,
    CF_PUSH_ELSE_R600	= 316,
    CF_TC_EG	= 317,
    CF_TC_R600	= 318,
    CF_VC_EG	= 319,
    CF_VC_R600	= 320,
    CNDE_INT	= 321,
    CNDE_eg	= 322,
    CNDE_r600	= 323,
    CNDGE_INT	= 324,
    CNDGE_eg	= 325,
    CNDGE_r600	= 326,
    CNDGT_INT	= 327,
    CNDGT_eg	= 328,
    CNDGT_r600	= 329,
    COS_cm	= 330,
    COS_eg	= 331,
    COS_r600	= 332,
    COS_r700	= 333,
    CUBE_eg_real	= 334,
    CUBE_r600_real	= 335,
    DOT4_eg	= 336,
    DOT4_r600	= 337,
    EG_ExportBuf	= 338,
    EG_ExportSwz	= 339,
    END_LOOP_EG	= 340,
    END_LOOP_R600	= 341,
    EXP_IEEE_cm	= 342,
    EXP_IEEE_eg	= 343,
    EXP_IEEE_r600	= 344,
    FETCH_CLAUSE	= 345,
    FFBH_UINT	= 346,
    FFBL_INT	= 347,
    FLOOR	= 348,
    FLT16_TO_FLT32	= 349,
    FLT32_TO_FLT16	= 350,
    FLT_TO_INT_eg	= 351,
    FLT_TO_INT_r600	= 352,
    FLT_TO_UINT_eg	= 353,
    FLT_TO_UINT_r600	= 354,
    FMA_eg	= 355,
    FRACT	= 356,
    GROUP_BARRIER	= 357,
    INTERP_LOAD_P0	= 358,
    INTERP_PAIR_XY	= 359,
    INTERP_PAIR_ZW	= 360,
    INTERP_VEC_LOAD	= 361,
    INTERP_XY	= 362,
    INTERP_ZW	= 363,
    INT_TO_FLT_eg	= 364,
    INT_TO_FLT_r600	= 365,
    KILLGT	= 366,
    LDS_ADD	= 367,
    LDS_ADD_RET	= 368,
    LDS_AND	= 369,
    LDS_AND_RET	= 370,
    LDS_BYTE_READ_RET	= 371,
    LDS_BYTE_WRITE	= 372,
    LDS_CMPST	= 373,
    LDS_CMPST_RET	= 374,
    LDS_MAX_INT	= 375,
    LDS_MAX_INT_RET	= 376,
    LDS_MAX_UINT	= 377,
    LDS_MAX_UINT_RET	= 378,
    LDS_MIN_INT	= 379,
    LDS_MIN_INT_RET	= 380,
    LDS_MIN_UINT	= 381,
    LDS_MIN_UINT_RET	= 382,
    LDS_OR	= 383,
    LDS_OR_RET	= 384,
    LDS_READ_RET	= 385,
    LDS_SHORT_READ_RET	= 386,
    LDS_SHORT_WRITE	= 387,
    LDS_SUB	= 388,
    LDS_SUB_RET	= 389,
    LDS_UBYTE_READ_RET	= 390,
    LDS_USHORT_READ_RET	= 391,
    LDS_WRITE	= 392,
    LDS_WRXCHG	= 393,
    LDS_WRXCHG_RET	= 394,
    LDS_XOR	= 395,
    LDS_XOR_RET	= 396,
    LITERALS	= 397,
    LOG_CLAMPED_eg	= 398,
    LOG_CLAMPED_r600	= 399,
    LOG_IEEE_cm	= 400,
    LOG_IEEE_eg	= 401,
    LOG_IEEE_r600	= 402,
    LOOP_BREAK_EG	= 403,
    LOOP_BREAK_R600	= 404,
    LSHL_eg	= 405,
    LSHL_r600	= 406,
    LSHR_eg	= 407,
    LSHR_r600	= 408,
    MAX	= 409,
    MAX_DX10	= 410,
    MAX_INT	= 411,
    MAX_UINT	= 412,
    MIN	= 413,
    MIN_DX10	= 414,
    MIN_INT	= 415,
    MIN_UINT	= 416,
    MOV	= 417,
    MOVA_INT_eg	= 418,
    MUL	= 419,
    MULADD_IEEE_eg	= 420,
    MULADD_IEEE_r600	= 421,
    MULADD_INT24_cm	= 422,
    MULADD_UINT24_eg	= 423,
    MULADD_eg	= 424,
    MULADD_r600	= 425,
    MULHI_INT_cm	= 426,
    MULHI_INT_cm24	= 427,
    MULHI_INT_eg	= 428,
    MULHI_INT_r600	= 429,
    MULHI_UINT24_eg	= 430,
    MULHI_UINT_cm	= 431,
    MULHI_UINT_cm24	= 432,
    MULHI_UINT_eg	= 433,
    MULHI_UINT_r600	= 434,
    MULLO_INT_cm	= 435,
    MULLO_INT_eg	= 436,
    MULLO_INT_r600	= 437,
    MULLO_UINT_cm	= 438,
    MULLO_UINT_eg	= 439,
    MULLO_UINT_r600	= 440,
    MUL_IEEE	= 441,
    MUL_INT24_cm	= 442,
    MUL_LIT_eg	= 443,
    MUL_LIT_r600	= 444,
    MUL_UINT24_eg	= 445,
    NOT_INT	= 446,
    OR_INT	= 447,
    PAD	= 448,
    POP_EG	= 449,
    POP_R600	= 450,
    PRED_SETE	= 451,
    PRED_SETE_INT	= 452,
    PRED_SETGE	= 453,
    PRED_SETGE_INT	= 454,
    PRED_SETGT	= 455,
    PRED_SETGT_INT	= 456,
    PRED_SETNE	= 457,
    PRED_SETNE_INT	= 458,
    R600_ExportBuf	= 459,
    R600_ExportSwz	= 460,
    RAT_ATOMIC_ADD_NORET	= 461,
    RAT_ATOMIC_ADD_RTN	= 462,
    RAT_ATOMIC_AND_NORET	= 463,
    RAT_ATOMIC_AND_RTN	= 464,
    RAT_ATOMIC_CMPXCHG_INT_NORET	= 465,
    RAT_ATOMIC_CMPXCHG_INT_RTN	= 466,
    RAT_ATOMIC_DEC_UINT_NORET	= 467,
    RAT_ATOMIC_DEC_UINT_RTN	= 468,
    RAT_ATOMIC_INC_UINT_NORET	= 469,
    RAT_ATOMIC_INC_UINT_RTN	= 470,
    RAT_ATOMIC_MAX_INT_NORET	= 471,
    RAT_ATOMIC_MAX_INT_RTN	= 472,
    RAT_ATOMIC_MAX_UINT_NORET	= 473,
    RAT_ATOMIC_MAX_UINT_RTN	= 474,
    RAT_ATOMIC_MIN_INT_NORET	= 475,
    RAT_ATOMIC_MIN_INT_RTN	= 476,
    RAT_ATOMIC_MIN_UINT_NORET	= 477,
    RAT_ATOMIC_MIN_UINT_RTN	= 478,
    RAT_ATOMIC_OR_NORET	= 479,
    RAT_ATOMIC_OR_RTN	= 480,
    RAT_ATOMIC_RSUB_NORET	= 481,
    RAT_ATOMIC_RSUB_RTN	= 482,
    RAT_ATOMIC_SUB_NORET	= 483,
    RAT_ATOMIC_SUB_RTN	= 484,
    RAT_ATOMIC_XCHG_INT_NORET	= 485,
    RAT_ATOMIC_XCHG_INT_RTN	= 486,
    RAT_ATOMIC_XOR_NORET	= 487,
    RAT_ATOMIC_XOR_RTN	= 488,
    RAT_MSKOR	= 489,
    RAT_STORE_DWORD128	= 490,
    RAT_STORE_DWORD32	= 491,
    RAT_STORE_DWORD64	= 492,
    RAT_STORE_TYPED_cm	= 493,
    RAT_STORE_TYPED_eg	= 494,
    RAT_WRITE_CACHELESS_128_eg	= 495,
    RAT_WRITE_CACHELESS_32_eg	= 496,
    RAT_WRITE_CACHELESS_64_eg	= 497,
    RECIPSQRT_CLAMPED_cm	= 498,
    RECIPSQRT_CLAMPED_eg	= 499,
    RECIPSQRT_CLAMPED_r600	= 500,
    RECIPSQRT_IEEE_cm	= 501,
    RECIPSQRT_IEEE_eg	= 502,
    RECIPSQRT_IEEE_r600	= 503,
    RECIP_CLAMPED_cm	= 504,
    RECIP_CLAMPED_eg	= 505,
    RECIP_CLAMPED_r600	= 506,
    RECIP_IEEE_cm	= 507,
    RECIP_IEEE_eg	= 508,
    RECIP_IEEE_r600	= 509,
    RECIP_UINT_eg	= 510,
    RECIP_UINT_r600	= 511,
    RNDNE	= 512,
    SETE	= 513,
    SETE_DX10	= 514,
    SETE_INT	= 515,
    SETGE_DX10	= 516,
    SETGE_INT	= 517,
    SETGE_UINT	= 518,
    SETGT_DX10	= 519,
    SETGT_INT	= 520,
    SETGT_UINT	= 521,
    SETNE_DX10	= 522,
    SETNE_INT	= 523,
    SGE	= 524,
    SGT	= 525,
    SIN_cm	= 526,
    SIN_eg	= 527,
    SIN_r600	= 528,
    SIN_r700	= 529,
    SNE	= 530,
    SUBB_UINT	= 531,
    SUB_INT	= 532,
    TEX_GET_GRADIENTS_H	= 533,
    TEX_GET_GRADIENTS_V	= 534,
    TEX_GET_TEXTURE_RESINFO	= 535,
    TEX_LD	= 536,
    TEX_LDPTR	= 537,
    TEX_SAMPLE	= 538,
    TEX_SAMPLE_C	= 539,
    TEX_SAMPLE_C_G	= 540,
    TEX_SAMPLE_C_L	= 541,
    TEX_SAMPLE_C_LB	= 542,
    TEX_SAMPLE_G	= 543,
    TEX_SAMPLE_L	= 544,
    TEX_SAMPLE_LB	= 545,
    TEX_SET_GRADIENTS_H	= 546,
    TEX_SET_GRADIENTS_V	= 547,
    TEX_VTX_CONSTBUF	= 548,
    TEX_VTX_TEXBUF	= 549,
    TRUNC	= 550,
    UINT_TO_FLT_eg	= 551,
    UINT_TO_FLT_r600	= 552,
    VTX_READ_128_cm	= 553,
    VTX_READ_128_eg	= 554,
    VTX_READ_16_cm	= 555,
    VTX_READ_16_eg	= 556,
    VTX_READ_32_cm	= 557,
    VTX_READ_32_eg	= 558,
    VTX_READ_64_cm	= 559,
    VTX_READ_64_eg	= 560,
    VTX_READ_8_cm	= 561,
    VTX_READ_8_eg	= 562,
    WHILE_LOOP_EG	= 563,
    WHILE_LOOP_R600	= 564,
    XOR_INT	= 565,
    INSTRUCTION_LIST_END = 566
  };

} // end namespace R600
} // end namespace llvm
#endif // GET_INSTRINFO_ENUM

#ifdef GET_INSTRINFO_SCHED_ENUM
#undef GET_INSTRINFO_SCHED_ENUM
namespace llvm {

namespace R600 {
namespace Sched {
  enum {
    NoInstrModel	= 0,
    NullALU	= 1,
    VecALU	= 2,
    AnyALU	= 3,
    TransALU	= 4,
    XALU	= 5,
    SCHED_LIST_END = 6
  };
} // end namespace Sched
} // end namespace R600
} // end namespace llvm
#endif // GET_INSTRINFO_SCHED_ENUM

#ifdef GET_INSTRINFO_MC_DESC
#undef GET_INSTRINFO_MC_DESC
namespace llvm {


static const MCOperandInfo OperandInfo2[] = { { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo3[] = { { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, };
static const MCOperandInfo OperandInfo4[] = { { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, };
static const MCOperandInfo OperandInfo5[] = { { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, ((0 << 16) | (1 << MCOI::TIED_TO)) }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, };
static const MCOperandInfo OperandInfo6[] = { { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, };
static const MCOperandInfo OperandInfo7[] = { { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo8[] = { { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, };
static const MCOperandInfo OperandInfo9[] = { { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, };
static const MCOperandInfo OperandInfo10[] = { { 0, 0|(1<<MCOI::LookupPtrRegClass), MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo11[] = { { 0, 0|(1<<MCOI::LookupPtrRegClass), MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, };
static const MCOperandInfo OperandInfo12[] = { { 0, 0|(1<<MCOI::LookupPtrRegClass), MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, };
static const MCOperandInfo OperandInfo13[] = { { 0, 0|(1<<MCOI::LookupPtrRegClass), MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo14[] = { { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { 0, 0|(1<<MCOI::LookupPtrRegClass), MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo15[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, };
static const MCOperandInfo OperandInfo16[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, };
static const MCOperandInfo OperandInfo17[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo18[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_IMM_0, 0 }, };
static const MCOperandInfo OperandInfo19[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, };
static const MCOperandInfo OperandInfo20[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_IMM_0, 0 }, };
static const MCOperandInfo OperandInfo21[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, };
static const MCOperandInfo OperandInfo22[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_2, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo23[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_2, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo24[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_2, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, };
static const MCOperandInfo OperandInfo25[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, };
static const MCOperandInfo OperandInfo26[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, };
static const MCOperandInfo OperandInfo27[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo28[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_IMM_0, 0 }, };
static const MCOperandInfo OperandInfo29[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, };
static const MCOperandInfo OperandInfo30[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, };
static const MCOperandInfo OperandInfo31[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, };
static const MCOperandInfo OperandInfo32[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, };
static const MCOperandInfo OperandInfo33[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_IMM_0, 0 }, };
static const MCOperandInfo OperandInfo34[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, };
static const MCOperandInfo OperandInfo35[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, };
static const MCOperandInfo OperandInfo36[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_2, 0 }, };
static const MCOperandInfo OperandInfo37[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_2, 0 }, };
static const MCOperandInfo OperandInfo38[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo39[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, };
static const MCOperandInfo OperandInfo40[] = { { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, };
static const MCOperandInfo OperandInfo41[] = { { -1, 0, MCOI::OPERAND_GENERIC_0, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_1, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_2, 0 }, { -1, 0, MCOI::OPERAND_GENERIC_IMM_0, 0 }, };
static const MCOperandInfo OperandInfo42[] = { { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, };
static const MCOperandInfo OperandInfo43[] = { { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, };
static const MCOperandInfo OperandInfo44[] = { { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, };
static const MCOperandInfo OperandInfo45[] = { { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, };
static const MCOperandInfo OperandInfo46[] = { { R600::R600_Reg128RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_Reg128RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, };
static const MCOperandInfo OperandInfo47[] = { { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_TReg32_XRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_TReg32_XRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_PredicateRegClassID, 0|(1<<MCOI::Predicate), MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_TReg32_YRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_TReg32_YRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_PredicateRegClassID, 0|(1<<MCOI::Predicate), MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_TReg32_ZRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_TReg32_ZRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_PredicateRegClassID, 0|(1<<MCOI::Predicate), MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_TReg32_WRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_TReg32_WRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_PredicateRegClassID, 0|(1<<MCOI::Predicate), MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo48[] = { { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_Predicate_BitRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, };
static const MCOperandInfo OperandInfo49[] = { { R600::R600_Predicate_BitRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, };
static const MCOperandInfo OperandInfo50[] = { { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_Reg64VerticalRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, };
static const MCOperandInfo OperandInfo51[] = { { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_Reg128VerticalRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, };
static const MCOperandInfo OperandInfo52[] = { { R600::R600_Reg64VerticalRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_Reg64VerticalRegClassID, 0, MCOI::OPERAND_REGISTER, ((0 << 16) | (1 << MCOI::TIED_TO)) }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, };
static const MCOperandInfo OperandInfo53[] = { { R600::R600_Reg128VerticalRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_Reg128VerticalRegClassID, 0, MCOI::OPERAND_REGISTER, ((0 << 16) | (1 << MCOI::TIED_TO)) }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, };
static const MCOperandInfo OperandInfo54[] = { { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, };
static const MCOperandInfo OperandInfo55[] = { { R600::R600_Reg128RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_Reg128RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_Reg128RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_Reg128RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, };
static const MCOperandInfo OperandInfo56[] = { { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_PredicateRegClassID, 0|(1<<MCOI::Predicate), MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo57[] = { { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_PredicateRegClassID, 0|(1<<MCOI::Predicate), MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo58[] = { { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_PredicateRegClassID, 0|(1<<MCOI::Predicate), MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo59[] = { { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, };
static const MCOperandInfo OperandInfo60[] = { { R600::R600_Reg128RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, };
static const MCOperandInfo OperandInfo61[] = { { R600::R600_Reg128RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, };
static const MCOperandInfo OperandInfo62[] = { { R600::R600_TReg32_XRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_TReg32_YRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { R600::R600_TReg32_YRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_TReg32_XRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, };
static const MCOperandInfo OperandInfo63[] = { { R600::R600_TReg32_ZRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_TReg32_WRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { R600::R600_TReg32_YRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_TReg32_XRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, };
static const MCOperandInfo OperandInfo64[] = { { R600::R600_Reg128RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, };
static const MCOperandInfo OperandInfo65[] = { { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_PredicateRegClassID, 0|(1<<MCOI::Predicate), MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo66[] = { { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_PredicateRegClassID, 0|(1<<MCOI::Predicate), MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo67[] = { { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_PredicateRegClassID, 0|(1<<MCOI::Predicate), MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo68[] = { { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_PredicateRegClassID, 0|(1<<MCOI::Predicate), MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo69[] = { { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_Reg32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { R600::R600_PredicateRegClassID, 0|(1<<MCOI::Predicate), MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo70[] = { { R600::R600_Reg128RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_Reg128RegClassID, 0, MCOI::OPERAND_REGISTER, ((0 << 16) | (1 << MCOI::TIED_TO)) }, { R600::R600_TReg32_XRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, };
static const MCOperandInfo OperandInfo71[] = { { R600::R600_Reg128RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_TReg32_XRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, };
static const MCOperandInfo OperandInfo72[] = { { R600::R600_TReg32_XRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_TReg32_XRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, };
static const MCOperandInfo OperandInfo73[] = { { R600::R600_Reg64RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_TReg32_XRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, };
static const MCOperandInfo OperandInfo74[] = { { R600::R600_Reg128RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_Reg128RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo75[] = { { R600::R600_Reg128RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_TReg32_XRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo76[] = { { R600::R600_TReg32_XRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_TReg32_XRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo77[] = { { R600::R600_Reg64RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_TReg32_XRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo78[] = { { R600::R600_Reg128RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_Reg128RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, };
static const MCOperandInfo OperandInfo79[] = { { R600::R600_Reg128RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_TReg32_XRegClassID, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, };
static const MCOperandInfo OperandInfo80[] = { { R600::R600_TReg32_XRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_TReg32_XRegClassID, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, };
static const MCOperandInfo OperandInfo81[] = { { R600::R600_TReg32_XRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_TReg32_XRegClassID, 0, MCOI::OPERAND_UNKNOWN, ((0 << 16) | (1 << MCOI::TIED_TO)) }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, };
static const MCOperandInfo OperandInfo82[] = { { R600::R600_Reg64RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { R600::R600_TReg32_XRegClassID, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, };

extern const MCInstrDesc R600Insts[] = {
  { 0,	1,	1,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Variadic), 0x0ULL, nullptr, nullptr, OperandInfo2 },  // Inst #0 = PHI
  { 1,	0,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Variadic), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #1 = INLINEASM
  { 2,	0,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Variadic)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #2 = INLINEASM_BR
  { 3,	1,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::NotDuplicable), 0x0ULL, nullptr, nullptr, OperandInfo3 },  // Inst #3 = CFI_INSTRUCTION
  { 4,	1,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::NotDuplicable), 0x0ULL, nullptr, nullptr, OperandInfo3 },  // Inst #4 = EH_LABEL
  { 5,	1,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::NotDuplicable), 0x0ULL, nullptr, nullptr, OperandInfo3 },  // Inst #5 = GC_LABEL
  { 6,	1,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::NotDuplicable), 0x0ULL, nullptr, nullptr, OperandInfo3 },  // Inst #6 = ANNOTATION_LABEL
  { 7,	0,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Variadic), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #7 = KILL
  { 8,	3,	1,	0,	0,	0|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo4 },  // Inst #8 = EXTRACT_SUBREG
  { 9,	4,	1,	0,	0,	0|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo5 },  // Inst #9 = INSERT_SUBREG
  { 10,	1,	1,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Rematerializable)|(1ULL<<MCID::CheapAsAMove), 0x0ULL, nullptr, nullptr, OperandInfo2 },  // Inst #10 = IMPLICIT_DEF
  { 11,	4,	1,	0,	0,	0|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo6 },  // Inst #11 = SUBREG_TO_REG
  { 12,	3,	1,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::CheapAsAMove), 0x0ULL, nullptr, nullptr, OperandInfo4 },  // Inst #12 = COPY_TO_REGCLASS
  { 13,	0,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Variadic), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #13 = DBG_VALUE
  { 14,	0,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Variadic), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #14 = DBG_INSTR_REF
  { 15,	1,	0,	0,	0,	0|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo2 },  // Inst #15 = DBG_LABEL
  { 16,	2,	1,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Variadic)|(1ULL<<MCID::CheapAsAMove), 0x0ULL, nullptr, nullptr, OperandInfo7 },  // Inst #16 = REG_SEQUENCE
  { 17,	2,	1,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::CheapAsAMove), 0x0ULL, nullptr, nullptr, OperandInfo7 },  // Inst #17 = COPY
  { 18,	0,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Variadic), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #18 = BUNDLE
  { 19,	1,	0,	0,	0,	0|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo3 },  // Inst #19 = LIFETIME_START
  { 20,	1,	0,	0,	0,	0|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo3 },  // Inst #20 = LIFETIME_END
  { 21,	2,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Call)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::Variadic)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo8 },  // Inst #21 = STACKMAP
  { 22,	0,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #22 = FENTRY_CALL
  { 23,	6,	1,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Call)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::Variadic)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo9 },  // Inst #23 = PATCHPOINT
  { 24,	1,	1,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::Rematerializable), 0x0ULL, nullptr, nullptr, OperandInfo10 },  // Inst #24 = LOAD_STACK_GUARD
  { 25,	1,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo3 },  // Inst #25 = PREALLOCATED_SETUP
  { 26,	3,	1,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo11 },  // Inst #26 = PREALLOCATED_ARG
  { 27,	0,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Call)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::Variadic)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #27 = STATEPOINT
  { 28,	2,	0,	0,	0,	0|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo12 },  // Inst #28 = LOCAL_ESCAPE
  { 29,	1,	1,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Branch)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::Terminator)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::Variadic)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo2 },  // Inst #29 = FAULTING_OP
  { 30,	0,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::Variadic)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #30 = PATCHABLE_OP
  { 31,	0,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #31 = PATCHABLE_FUNCTION_ENTER
  { 32,	0,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Return)|(1ULL<<MCID::Terminator)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::Variadic)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #32 = PATCHABLE_RET
  { 33,	0,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #33 = PATCHABLE_FUNCTION_EXIT
  { 34,	0,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Return)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::Variadic)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #34 = PATCHABLE_TAIL_CALL
  { 35,	2,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Call)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo13 },  // Inst #35 = PATCHABLE_EVENT_CALL
  { 36,	3,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Call)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo14 },  // Inst #36 = PATCHABLE_TYPED_EVENT_CALL
  { 37,	0,	0,	0,	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Variadic)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #37 = ICALL_BRANCH_FUNNEL
  { 38,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #38 = G_ADD
  { 39,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #39 = G_SUB
  { 40,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #40 = G_MUL
  { 41,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #41 = G_SDIV
  { 42,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #42 = G_UDIV
  { 43,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #43 = G_SREM
  { 44,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #44 = G_UREM
  { 45,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #45 = G_AND
  { 46,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #46 = G_OR
  { 47,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #47 = G_XOR
  { 48,	1,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo16 },  // Inst #48 = G_IMPLICIT_DEF
  { 49,	1,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Variadic), 0x0ULL, nullptr, nullptr, OperandInfo16 },  // Inst #49 = G_PHI
  { 50,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo17 },  // Inst #50 = G_FRAME_INDEX
  { 51,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo17 },  // Inst #51 = G_GLOBAL_VALUE
  { 52,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo18 },  // Inst #52 = G_EXTRACT
  { 53,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Variadic), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #53 = G_UNMERGE_VALUES
  { 54,	4,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo20 },  // Inst #54 = G_INSERT
  { 55,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Variadic), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #55 = G_MERGE_VALUES
  { 56,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Variadic), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #56 = G_BUILD_VECTOR
  { 57,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Variadic), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #57 = G_BUILD_VECTOR_TRUNC
  { 58,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Variadic), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #58 = G_CONCAT_VECTORS
  { 59,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #59 = G_PTRTOINT
  { 60,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #60 = G_INTTOPTR
  { 61,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #61 = G_BITCAST
  { 62,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #62 = G_FREEZE
  { 63,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #63 = G_INTRINSIC_TRUNC
  { 64,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #64 = G_INTRINSIC_ROUND
  { 65,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #65 = G_INTRINSIC_LRINT
  { 66,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #66 = G_INTRINSIC_ROUNDEVEN
  { 67,	1,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo16 },  // Inst #67 = G_READCYCLECOUNTER
  { 68,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #68 = G_LOAD
  { 69,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #69 = G_SEXTLOAD
  { 70,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #70 = G_ZEXTLOAD
  { 71,	5,	2,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad), 0x0ULL, nullptr, nullptr, OperandInfo22 },  // Inst #71 = G_INDEXED_LOAD
  { 72,	5,	2,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad), 0x0ULL, nullptr, nullptr, OperandInfo22 },  // Inst #72 = G_INDEXED_SEXTLOAD
  { 73,	5,	2,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad), 0x0ULL, nullptr, nullptr, OperandInfo22 },  // Inst #73 = G_INDEXED_ZEXTLOAD
  { 74,	2,	0,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #74 = G_STORE
  { 75,	5,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo23 },  // Inst #75 = G_INDEXED_STORE
  { 76,	5,	2,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo24 },  // Inst #76 = G_ATOMIC_CMPXCHG_WITH_SUCCESS
  { 77,	4,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo25 },  // Inst #77 = G_ATOMIC_CMPXCHG
  { 78,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo26 },  // Inst #78 = G_ATOMICRMW_XCHG
  { 79,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo26 },  // Inst #79 = G_ATOMICRMW_ADD
  { 80,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo26 },  // Inst #80 = G_ATOMICRMW_SUB
  { 81,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo26 },  // Inst #81 = G_ATOMICRMW_AND
  { 82,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo26 },  // Inst #82 = G_ATOMICRMW_NAND
  { 83,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo26 },  // Inst #83 = G_ATOMICRMW_OR
  { 84,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo26 },  // Inst #84 = G_ATOMICRMW_XOR
  { 85,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo26 },  // Inst #85 = G_ATOMICRMW_MAX
  { 86,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo26 },  // Inst #86 = G_ATOMICRMW_MIN
  { 87,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo26 },  // Inst #87 = G_ATOMICRMW_UMAX
  { 88,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo26 },  // Inst #88 = G_ATOMICRMW_UMIN
  { 89,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo26 },  // Inst #89 = G_ATOMICRMW_FADD
  { 90,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo26 },  // Inst #90 = G_ATOMICRMW_FSUB
  { 91,	2,	0,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo8 },  // Inst #91 = G_FENCE
  { 92,	2,	0,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Branch)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo17 },  // Inst #92 = G_BRCOND
  { 93,	1,	0,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Branch)|(1ULL<<MCID::IndirectBranch)|(1ULL<<MCID::Barrier)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo16 },  // Inst #93 = G_BRINDIRECT
  { 94,	1,	0,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Variadic)|(1ULL<<MCID::Convergent), 0x0ULL, nullptr, nullptr, OperandInfo2 },  // Inst #94 = G_INTRINSIC
  { 95,	1,	0,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::Variadic)|(1ULL<<MCID::UnmodeledSideEffects)|(1ULL<<MCID::Convergent), 0x0ULL, nullptr, nullptr, OperandInfo2 },  // Inst #95 = G_INTRINSIC_W_SIDE_EFFECTS
  { 96,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #96 = G_ANYEXT
  { 97,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #97 = G_TRUNC
  { 98,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo17 },  // Inst #98 = G_CONSTANT
  { 99,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo17 },  // Inst #99 = G_FCONSTANT
  { 100,	1,	0,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo16 },  // Inst #100 = G_VASTART
  { 101,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo27 },  // Inst #101 = G_VAARG
  { 102,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #102 = G_SEXT
  { 103,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo28 },  // Inst #103 = G_SEXT_INREG
  { 104,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #104 = G_ZEXT
  { 105,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo29 },  // Inst #105 = G_SHL
  { 106,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo29 },  // Inst #106 = G_LSHR
  { 107,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo29 },  // Inst #107 = G_ASHR
  { 108,	4,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo30 },  // Inst #108 = G_FSHL
  { 109,	4,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo30 },  // Inst #109 = G_FSHR
  { 110,	4,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo31 },  // Inst #110 = G_ICMP
  { 111,	4,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo31 },  // Inst #111 = G_FCMP
  { 112,	4,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo25 },  // Inst #112 = G_SELECT
  { 113,	4,	2,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo25 },  // Inst #113 = G_UADDO
  { 114,	5,	2,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo32 },  // Inst #114 = G_UADDE
  { 115,	4,	2,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo25 },  // Inst #115 = G_USUBO
  { 116,	5,	2,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo32 },  // Inst #116 = G_USUBE
  { 117,	4,	2,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo25 },  // Inst #117 = G_SADDO
  { 118,	5,	2,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo32 },  // Inst #118 = G_SADDE
  { 119,	4,	2,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo25 },  // Inst #119 = G_SSUBO
  { 120,	5,	2,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo32 },  // Inst #120 = G_SSUBE
  { 121,	4,	2,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo25 },  // Inst #121 = G_UMULO
  { 122,	4,	2,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo25 },  // Inst #122 = G_SMULO
  { 123,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #123 = G_UMULH
  { 124,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #124 = G_SMULH
  { 125,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #125 = G_UADDSAT
  { 126,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #126 = G_SADDSAT
  { 127,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #127 = G_USUBSAT
  { 128,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #128 = G_SSUBSAT
  { 129,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo29 },  // Inst #129 = G_USHLSAT
  { 130,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo29 },  // Inst #130 = G_SSHLSAT
  { 131,	4,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo33 },  // Inst #131 = G_SMULFIX
  { 132,	4,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo33 },  // Inst #132 = G_UMULFIX
  { 133,	4,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo33 },  // Inst #133 = G_SMULFIXSAT
  { 134,	4,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo33 },  // Inst #134 = G_UMULFIXSAT
  { 135,	4,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo33 },  // Inst #135 = G_SDIVFIX
  { 136,	4,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo33 },  // Inst #136 = G_UDIVFIX
  { 137,	4,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo33 },  // Inst #137 = G_SDIVFIXSAT
  { 138,	4,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo33 },  // Inst #138 = G_UDIVFIXSAT
  { 139,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #139 = G_FADD
  { 140,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #140 = G_FSUB
  { 141,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #141 = G_FMUL
  { 142,	4,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo34 },  // Inst #142 = G_FMA
  { 143,	4,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo34 },  // Inst #143 = G_FMAD
  { 144,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #144 = G_FDIV
  { 145,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #145 = G_FREM
  { 146,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #146 = G_FPOW
  { 147,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo29 },  // Inst #147 = G_FPOWI
  { 148,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #148 = G_FEXP
  { 149,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #149 = G_FEXP2
  { 150,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #150 = G_FLOG
  { 151,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #151 = G_FLOG2
  { 152,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #152 = G_FLOG10
  { 153,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #153 = G_FNEG
  { 154,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #154 = G_FPEXT
  { 155,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #155 = G_FPTRUNC
  { 156,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #156 = G_FPTOSI
  { 157,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #157 = G_FPTOUI
  { 158,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #158 = G_SITOFP
  { 159,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #159 = G_UITOFP
  { 160,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #160 = G_FABS
  { 161,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo29 },  // Inst #161 = G_FCOPYSIGN
  { 162,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #162 = G_FCANONICALIZE
  { 163,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #163 = G_FMINNUM
  { 164,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #164 = G_FMAXNUM
  { 165,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #165 = G_FMINNUM_IEEE
  { 166,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #166 = G_FMAXNUM_IEEE
  { 167,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #167 = G_FMINIMUM
  { 168,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #168 = G_FMAXIMUM
  { 169,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo29 },  // Inst #169 = G_PTR_ADD
  { 170,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo29 },  // Inst #170 = G_PTRMASK
  { 171,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #171 = G_SMIN
  { 172,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #172 = G_SMAX
  { 173,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #173 = G_UMIN
  { 174,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Commutable), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #174 = G_UMAX
  { 175,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #175 = G_ABS
  { 176,	1,	0,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Branch)|(1ULL<<MCID::Barrier)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo2 },  // Inst #176 = G_BR
  { 177,	3,	0,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Branch)|(1ULL<<MCID::IndirectBranch)|(1ULL<<MCID::Barrier)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo35 },  // Inst #177 = G_BRJT
  { 178,	4,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo36 },  // Inst #178 = G_INSERT_VECTOR_ELT
  { 179,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo37 },  // Inst #179 = G_EXTRACT_VECTOR_ELT
  { 180,	4,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo38 },  // Inst #180 = G_SHUFFLE_VECTOR
  { 181,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #181 = G_CTTZ
  { 182,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #182 = G_CTTZ_ZERO_UNDEF
  { 183,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #183 = G_CTLZ
  { 184,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #184 = G_CTLZ_ZERO_UNDEF
  { 185,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #185 = G_CTPOP
  { 186,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #186 = G_BSWAP
  { 187,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #187 = G_BITREVERSE
  { 188,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #188 = G_FCEIL
  { 189,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #189 = G_FCOS
  { 190,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #190 = G_FSIN
  { 191,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #191 = G_FSQRT
  { 192,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #192 = G_FFLOOR
  { 193,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #193 = G_FRINT
  { 194,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #194 = G_FNEARBYINT
  { 195,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #195 = G_ADDRSPACE_CAST
  { 196,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo17 },  // Inst #196 = G_BLOCK_ADDR
  { 197,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo17 },  // Inst #197 = G_JUMP_TABLE
  { 198,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo39 },  // Inst #198 = G_DYN_STACKALLOC
  { 199,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayRaiseFPException)|(1ULL<<MCID::Commutable)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #199 = G_STRICT_FADD
  { 200,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayRaiseFPException)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #200 = G_STRICT_FSUB
  { 201,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayRaiseFPException)|(1ULL<<MCID::Commutable)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #201 = G_STRICT_FMUL
  { 202,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayRaiseFPException)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #202 = G_STRICT_FDIV
  { 203,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayRaiseFPException)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo15 },  // Inst #203 = G_STRICT_FREM
  { 204,	4,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayRaiseFPException)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo34 },  // Inst #204 = G_STRICT_FMA
  { 205,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayRaiseFPException)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo21 },  // Inst #205 = G_STRICT_FSQRT
  { 206,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::UnmodeledSideEffects)|(1ULL<<MCID::Convergent), 0x0ULL, nullptr, nullptr, OperandInfo17 },  // Inst #206 = G_READ_REGISTER
  { 207,	2,	0,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::UnmodeledSideEffects)|(1ULL<<MCID::Convergent), 0x0ULL, nullptr, nullptr, OperandInfo40 },  // Inst #207 = G_WRITE_REGISTER
  { 208,	4,	0,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo41 },  // Inst #208 = G_MEMCPY
  { 209,	4,	0,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo41 },  // Inst #209 = G_MEMMOVE
  { 210,	4,	0,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayStore), 0x0ULL, nullptr, nullptr, OperandInfo41 },  // Inst #210 = G_MEMSET
  { 211,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo37 },  // Inst #211 = G_VECREDUCE_SEQ_FADD
  { 212,	3,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo37 },  // Inst #212 = G_VECREDUCE_SEQ_FMUL
  { 213,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #213 = G_VECREDUCE_FADD
  { 214,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #214 = G_VECREDUCE_FMUL
  { 215,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #215 = G_VECREDUCE_FMAX
  { 216,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #216 = G_VECREDUCE_FMIN
  { 217,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #217 = G_VECREDUCE_ADD
  { 218,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #218 = G_VECREDUCE_MUL
  { 219,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #219 = G_VECREDUCE_AND
  { 220,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #220 = G_VECREDUCE_OR
  { 221,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #221 = G_VECREDUCE_XOR
  { 222,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #222 = G_VECREDUCE_SMAX
  { 223,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #223 = G_VECREDUCE_SMIN
  { 224,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #224 = G_VECREDUCE_UMAX
  { 225,	2,	1,	0,	0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo19 },  // Inst #225 = G_VECREDUCE_UMIN
  { 226,	1,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Branch)|(1ULL<<MCID::Barrier)|(1ULL<<MCID::Terminator)|(1ULL<<MCID::UsesCustomInserter), 0x0ULL, nullptr, nullptr, OperandInfo2 },  // Inst #226 = BRANCH
  { 227,	2,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Branch)|(1ULL<<MCID::Barrier)|(1ULL<<MCID::Terminator)|(1ULL<<MCID::UsesCustomInserter), 0x0ULL, nullptr, nullptr, OperandInfo42 },  // Inst #227 = BRANCH_COND_f32
  { 228,	2,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Branch)|(1ULL<<MCID::Barrier)|(1ULL<<MCID::Terminator)|(1ULL<<MCID::UsesCustomInserter), 0x0ULL, nullptr, nullptr, OperandInfo42 },  // Inst #228 = BRANCH_COND_i32
  { 229,	0,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #229 = BREAK
  { 230,	2,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo43 },  // Inst #230 = BREAKC_f32
  { 231,	2,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo43 },  // Inst #231 = BREAKC_i32
  { 232,	1,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo44 },  // Inst #232 = BREAK_LOGICALNZ_f32
  { 233,	1,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo44 },  // Inst #233 = BREAK_LOGICALNZ_i32
  { 234,	1,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo44 },  // Inst #234 = BREAK_LOGICALZ_f32
  { 235,	1,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo44 },  // Inst #235 = BREAK_LOGICALZ_i32
  { 236,	2,	1,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::CheapAsAMove), 0x0ULL, nullptr, nullptr, OperandInfo45 },  // Inst #236 = CONST_COPY
  { 237,	0,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #237 = CONTINUE
  { 238,	2,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo43 },  // Inst #238 = CONTINUEC_f32
  { 239,	2,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo43 },  // Inst #239 = CONTINUEC_i32
  { 240,	1,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo44 },  // Inst #240 = CONTINUE_LOGICALNZ_f32
  { 241,	1,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo44 },  // Inst #241 = CONTINUE_LOGICALNZ_i32
  { 242,	1,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo44 },  // Inst #242 = CONTINUE_LOGICALZ_f32
  { 243,	1,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo44 },  // Inst #243 = CONTINUE_LOGICALZ_i32
  { 244,	2,	1,	0,	2,	0|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo46 },  // Inst #244 = CUBE_eg_pseudo
  { 245,	2,	1,	0,	2,	0|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo46 },  // Inst #245 = CUBE_r600_pseudo
  { 246,	0,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #246 = DEFAULT
  { 247,	71,	1,	0,	3,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Predicable), 0x0ULL, nullptr, nullptr, OperandInfo47 },  // Inst #247 = DOT_4
  { 248,	0,	0,	0,	1,	0|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #248 = DUMMY_CHAIN
  { 249,	0,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #249 = ELSE
  { 250,	0,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #250 = END
  { 251,	0,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #251 = ENDFUNC
  { 252,	0,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #252 = ENDIF
  { 253,	0,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #253 = ENDLOOP
  { 254,	0,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #254 = ENDMAIN
  { 255,	0,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #255 = ENDSWITCH
  { 256,	2,	1,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::UsesCustomInserter), 0x0ULL, nullptr, nullptr, OperandInfo43 },  // Inst #256 = FABS_R600
  { 257,	2,	1,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::UsesCustomInserter), 0x0ULL, nullptr, nullptr, OperandInfo43 },  // Inst #257 = FNEG_R600
  { 258,	0,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #258 = FUNC
  { 259,	2,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo43 },  // Inst #259 = IFC_f32
  { 260,	2,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo43 },  // Inst #260 = IFC_i32
  { 261,	1,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo44 },  // Inst #261 = IF_LOGICALNZ_f32
  { 262,	1,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo44 },  // Inst #262 = IF_LOGICALNZ_i32
  { 263,	1,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo44 },  // Inst #263 = IF_LOGICALZ_f32
  { 264,	1,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, OperandInfo44 },  // Inst #264 = IF_LOGICALZ_i32
  { 265,	1,	0,	0,	1,	0|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo44 },  // Inst #265 = IF_PREDICATE_SET
  { 266,	1,	0,	0,	3,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Branch)|(1ULL<<MCID::Barrier)|(1ULL<<MCID::Predicable)|(1ULL<<MCID::Terminator)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo2 },  // Inst #266 = JUMP
  { 267,	2,	0,	0,	3,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Branch)|(1ULL<<MCID::Terminator)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo48 },  // Inst #267 = JUMP_COND
  { 268,	1,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo44 },  // Inst #268 = MASK_WRITE
  { 269,	2,	1,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::UsesCustomInserter), 0x0ULL, nullptr, nullptr, OperandInfo45 },  // Inst #269 = MOV_IMM_F32
  { 270,	2,	1,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::UsesCustomInserter), 0x0ULL, nullptr, nullptr, OperandInfo45 },  // Inst #270 = MOV_IMM_GLOBAL_ADDR
  { 271,	2,	1,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::UsesCustomInserter), 0x0ULL, nullptr, nullptr, OperandInfo45 },  // Inst #271 = MOV_IMM_I32
  { 272,	4,	1,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::UnmodeledSideEffects), 0x180ULL, nullptr, nullptr, OperandInfo49 },  // Inst #272 = PRED_X
  { 273,	3,	1,	0,	3,	0|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo50 },  // Inst #273 = R600_EXTRACT_ELT_V2
  { 274,	3,	1,	0,	3,	0|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo51 },  // Inst #274 = R600_EXTRACT_ELT_V4
  { 275,	4,	1,	0,	3,	0|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo52 },  // Inst #275 = R600_INSERT_ELT_V2
  { 276,	4,	1,	0,	3,	0|(1ULL<<MCID::Pseudo), 0x0ULL, nullptr, nullptr, OperandInfo53 },  // Inst #276 = R600_INSERT_ELT_V4
  { 277,	4,	1,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad), 0x8000000000000000ULL, nullptr, nullptr, OperandInfo54 },  // Inst #277 = R600_RegisterLoad
  { 278,	4,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayStore), 0x4000000000000000ULL, nullptr, nullptr, OperandInfo54 },  // Inst #278 = R600_RegisterStore
  { 279,	0,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #279 = RETDYN
  { 280,	0,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Return)|(1ULL<<MCID::Terminator)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::Variadic), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #280 = RETURN
  { 281,	7,	1,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::UnmodeledSideEffects), 0x2000ULL, nullptr, nullptr, OperandInfo55 },  // Inst #281 = TXD
  { 282,	7,	1,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::UnmodeledSideEffects), 0x2000ULL, nullptr, nullptr, OperandInfo55 },  // Inst #282 = TXD_SHADOW
  { 283,	0,	0,	0,	1,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Terminator), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #283 = WHILELOOP
  { 284,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #284 = ADD
  { 285,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #285 = ADDC_UINT
  { 286,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #286 = ADD_INT
  { 287,	1,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo3 },  // Inst #287 = ALU_CLAUSE
  { 288,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #288 = AND_INT
  { 289,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #289 = ASHR_eg
  { 290,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #290 = ASHR_r600
  { 291,	14,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #291 = BCNT_INT
  { 292,	19,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #292 = BFE_INT_eg
  { 293,	19,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #293 = BFE_UINT_eg
  { 294,	19,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #294 = BFI_INT_eg
  { 295,	21,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #295 = BFM_INT_eg
  { 296,	19,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #296 = BIT_ALIGN_INT_eg
  { 297,	14,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #297 = CEIL
  { 298,	9,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo59 },  // Inst #298 = CF_ALU
  { 299,	9,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo59 },  // Inst #299 = CF_ALU_BREAK
  { 300,	9,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo59 },  // Inst #300 = CF_ALU_CONTINUE
  { 301,	9,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo59 },  // Inst #301 = CF_ALU_ELSE_AFTER
  { 302,	9,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo59 },  // Inst #302 = CF_ALU_POP_AFTER
  { 303,	9,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo59 },  // Inst #303 = CF_ALU_PUSH_BEFORE
  { 304,	0,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #304 = CF_CALL_FS_EG
  { 305,	0,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #305 = CF_CALL_FS_R600
  { 306,	1,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo3 },  // Inst #306 = CF_CONTINUE_EG
  { 307,	1,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo3 },  // Inst #307 = CF_CONTINUE_R600
  { 308,	2,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo8 },  // Inst #308 = CF_ELSE_EG
  { 309,	2,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo8 },  // Inst #309 = CF_ELSE_R600
  { 310,	0,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #310 = CF_END_CM
  { 311,	0,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #311 = CF_END_EG
  { 312,	0,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #312 = CF_END_R600
  { 313,	2,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo8 },  // Inst #313 = CF_JUMP_EG
  { 314,	2,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo8 },  // Inst #314 = CF_JUMP_R600
  { 315,	2,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo8 },  // Inst #315 = CF_PUSH_EG
  { 316,	1,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo3 },  // Inst #316 = CF_PUSH_ELSE_R600
  { 317,	2,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo8 },  // Inst #317 = CF_TC_EG
  { 318,	2,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo8 },  // Inst #318 = CF_TC_R600
  { 319,	2,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo8 },  // Inst #319 = CF_VC_EG
  { 320,	2,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo8 },  // Inst #320 = CF_VC_R600
  { 321,	19,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #321 = CNDE_INT
  { 322,	19,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #322 = CNDE_eg
  { 323,	19,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #323 = CNDE_r600
  { 324,	19,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #324 = CNDGE_INT
  { 325,	19,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #325 = CNDGE_eg
  { 326,	19,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #326 = CNDGE_r600
  { 327,	19,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #327 = CNDGT_INT
  { 328,	19,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #328 = CNDGT_eg
  { 329,	19,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #329 = CNDGT_r600
  { 330,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4650ULL, nullptr, nullptr, OperandInfo57 },  // Inst #330 = COS_cm
  { 331,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4610ULL, nullptr, nullptr, OperandInfo57 },  // Inst #331 = COS_eg
  { 332,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4610ULL, nullptr, nullptr, OperandInfo57 },  // Inst #332 = COS_r600
  { 333,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4610ULL, nullptr, nullptr, OperandInfo57 },  // Inst #333 = COS_r700
  { 334,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #334 = CUBE_eg_real
  { 335,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #335 = CUBE_r600_real
  { 336,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #336 = DOT4_eg
  { 337,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #337 = DOT4_r600
  { 338,	7,	0,	0,	1,	0|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo60 },  // Inst #338 = EG_ExportBuf
  { 339,	9,	0,	0,	1,	0|(1ULL<<MCID::NotDuplicable)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo61 },  // Inst #339 = EG_ExportSwz
  { 340,	1,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo3 },  // Inst #340 = END_LOOP_EG
  { 341,	1,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo3 },  // Inst #341 = END_LOOP_R600
  { 342,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4640ULL, nullptr, nullptr, OperandInfo57 },  // Inst #342 = EXP_IEEE_cm
  { 343,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #343 = EXP_IEEE_eg
  { 344,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #344 = EXP_IEEE_r600
  { 345,	1,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo3 },  // Inst #345 = FETCH_CLAUSE
  { 346,	14,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #346 = FFBH_UINT
  { 347,	14,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #347 = FFBL_INT
  { 348,	14,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #348 = FLOOR
  { 349,	14,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #349 = FLT16_TO_FLT32
  { 350,	14,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #350 = FLT32_TO_FLT16
  { 351,	14,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #351 = FLT_TO_INT_eg
  { 352,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #352 = FLT_TO_INT_r600
  { 353,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #353 = FLT_TO_UINT_eg
  { 354,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #354 = FLT_TO_UINT_r600
  { 355,	19,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #355 = FMA_eg
  { 356,	14,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #356 = FRACT
  { 357,	0,	0,	0,	3,	0|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x4000ULL, nullptr, nullptr, nullptr },  // Inst #357 = GROUP_BARRIER
  { 358,	14,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #358 = INTERP_LOAD_P0
  { 359,	5,	2,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo62 },  // Inst #359 = INTERP_PAIR_XY
  { 360,	5,	2,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo63 },  // Inst #360 = INTERP_PAIR_ZW
  { 361,	2,	1,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo64 },  // Inst #361 = INTERP_VEC_LOAD
  { 362,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #362 = INTERP_XY
  { 363,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #363 = INTERP_ZW
  { 364,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #364 = INT_TO_FLT_eg
  { 365,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #365 = INT_TO_FLT_r600
  { 366,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UnmodeledSideEffects), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #366 = KILLGT
  { 367,	9,	0,	0,	5,	0|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UnmodeledSideEffects), 0x14200ULL, nullptr, nullptr, OperandInfo65 },  // Inst #367 = LDS_ADD
  { 368,	10,	1,	0,	5,	0|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UsesCustomInserter), 0x14200ULL, nullptr, nullptr, OperandInfo66 },  // Inst #368 = LDS_ADD_RET
  { 369,	9,	0,	0,	5,	0|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UnmodeledSideEffects), 0x14200ULL, nullptr, nullptr, OperandInfo65 },  // Inst #369 = LDS_AND
  { 370,	10,	1,	0,	5,	0|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UsesCustomInserter), 0x14200ULL, nullptr, nullptr, OperandInfo66 },  // Inst #370 = LDS_AND_RET
  { 371,	7,	1,	0,	5,	0|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UsesCustomInserter), 0xc200ULL, nullptr, nullptr, OperandInfo67 },  // Inst #371 = LDS_BYTE_READ_RET
  { 372,	9,	0,	0,	5,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::Predicable), 0x14200ULL, nullptr, nullptr, OperandInfo65 },  // Inst #372 = LDS_BYTE_WRITE
  { 373,	12,	0,	0,	5,	0|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UnmodeledSideEffects), 0x44200ULL, nullptr, nullptr, OperandInfo68 },  // Inst #373 = LDS_CMPST
  { 374,	13,	1,	0,	5,	0|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UsesCustomInserter), 0x44200ULL, nullptr, nullptr, OperandInfo69 },  // Inst #374 = LDS_CMPST_RET
  { 375,	9,	0,	0,	5,	0|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UnmodeledSideEffects), 0x14200ULL, nullptr, nullptr, OperandInfo65 },  // Inst #375 = LDS_MAX_INT
  { 376,	10,	1,	0,	5,	0|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UsesCustomInserter), 0x14200ULL, nullptr, nullptr, OperandInfo66 },  // Inst #376 = LDS_MAX_INT_RET
  { 377,	9,	0,	0,	5,	0|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UnmodeledSideEffects), 0x14200ULL, nullptr, nullptr, OperandInfo65 },  // Inst #377 = LDS_MAX_UINT
  { 378,	10,	1,	0,	5,	0|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UsesCustomInserter), 0x14200ULL, nullptr, nullptr, OperandInfo66 },  // Inst #378 = LDS_MAX_UINT_RET
  { 379,	9,	0,	0,	5,	0|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UnmodeledSideEffects), 0x14200ULL, nullptr, nullptr, OperandInfo65 },  // Inst #379 = LDS_MIN_INT
  { 380,	10,	1,	0,	5,	0|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UsesCustomInserter), 0x14200ULL, nullptr, nullptr, OperandInfo66 },  // Inst #380 = LDS_MIN_INT_RET
  { 381,	9,	0,	0,	5,	0|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UnmodeledSideEffects), 0x14200ULL, nullptr, nullptr, OperandInfo65 },  // Inst #381 = LDS_MIN_UINT
  { 382,	10,	1,	0,	5,	0|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UsesCustomInserter), 0x14200ULL, nullptr, nullptr, OperandInfo66 },  // Inst #382 = LDS_MIN_UINT_RET
  { 383,	9,	0,	0,	5,	0|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UnmodeledSideEffects), 0x14200ULL, nullptr, nullptr, OperandInfo65 },  // Inst #383 = LDS_OR
  { 384,	10,	1,	0,	5,	0|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UsesCustomInserter), 0x14200ULL, nullptr, nullptr, OperandInfo66 },  // Inst #384 = LDS_OR_RET
  { 385,	7,	1,	0,	5,	0|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UsesCustomInserter), 0xc200ULL, nullptr, nullptr, OperandInfo67 },  // Inst #385 = LDS_READ_RET
  { 386,	7,	1,	0,	5,	0|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UsesCustomInserter), 0xc200ULL, nullptr, nullptr, OperandInfo67 },  // Inst #386 = LDS_SHORT_READ_RET
  { 387,	9,	0,	0,	5,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::Predicable), 0x14200ULL, nullptr, nullptr, OperandInfo65 },  // Inst #387 = LDS_SHORT_WRITE
  { 388,	9,	0,	0,	5,	0|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UnmodeledSideEffects), 0x14200ULL, nullptr, nullptr, OperandInfo65 },  // Inst #388 = LDS_SUB
  { 389,	10,	1,	0,	5,	0|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UsesCustomInserter), 0x14200ULL, nullptr, nullptr, OperandInfo66 },  // Inst #389 = LDS_SUB_RET
  { 390,	7,	1,	0,	5,	0|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UsesCustomInserter), 0xc200ULL, nullptr, nullptr, OperandInfo67 },  // Inst #390 = LDS_UBYTE_READ_RET
  { 391,	7,	1,	0,	5,	0|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UsesCustomInserter), 0xc200ULL, nullptr, nullptr, OperandInfo67 },  // Inst #391 = LDS_USHORT_READ_RET
  { 392,	9,	0,	0,	5,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::Predicable), 0x14200ULL, nullptr, nullptr, OperandInfo65 },  // Inst #392 = LDS_WRITE
  { 393,	9,	0,	0,	5,	0|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UnmodeledSideEffects), 0x14200ULL, nullptr, nullptr, OperandInfo65 },  // Inst #393 = LDS_WRXCHG
  { 394,	10,	1,	0,	5,	0|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UsesCustomInserter), 0x14200ULL, nullptr, nullptr, OperandInfo66 },  // Inst #394 = LDS_WRXCHG_RET
  { 395,	9,	0,	0,	5,	0|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UnmodeledSideEffects), 0x14200ULL, nullptr, nullptr, OperandInfo65 },  // Inst #395 = LDS_XOR
  { 396,	10,	1,	0,	5,	0|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UsesCustomInserter), 0x14200ULL, nullptr, nullptr, OperandInfo66 },  // Inst #396 = LDS_XOR_RET
  { 397,	2,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo7 },  // Inst #397 = LITERALS
  { 398,	14,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #398 = LOG_CLAMPED_eg
  { 399,	14,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #399 = LOG_CLAMPED_r600
  { 400,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4640ULL, nullptr, nullptr, OperandInfo57 },  // Inst #400 = LOG_IEEE_cm
  { 401,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #401 = LOG_IEEE_eg
  { 402,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #402 = LOG_IEEE_r600
  { 403,	1,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo3 },  // Inst #403 = LOOP_BREAK_EG
  { 404,	1,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo3 },  // Inst #404 = LOOP_BREAK_R600
  { 405,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #405 = LSHL_eg
  { 406,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #406 = LSHL_r600
  { 407,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #407 = LSHR_eg
  { 408,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #408 = LSHR_r600
  { 409,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #409 = MAX
  { 410,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #410 = MAX_DX10
  { 411,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #411 = MAX_INT
  { 412,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #412 = MAX_UINT
  { 413,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #413 = MIN
  { 414,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #414 = MIN_DX10
  { 415,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #415 = MIN_INT
  { 416,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #416 = MIN_UINT
  { 417,	14,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #417 = MOV
  { 418,	14,	1,	0,	2,	0|(1ULL<<MCID::Predicable)|(1ULL<<MCID::UnmodeledSideEffects), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #418 = MOVA_INT_eg
  { 419,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #419 = MUL
  { 420,	19,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #420 = MULADD_IEEE_eg
  { 421,	19,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #421 = MULADD_IEEE_r600
  { 422,	19,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #422 = MULADD_INT24_cm
  { 423,	19,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #423 = MULADD_UINT24_eg
  { 424,	19,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #424 = MULADD_eg
  { 425,	19,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #425 = MULADD_r600
  { 426,	21,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4a40ULL, nullptr, nullptr, OperandInfo56 },  // Inst #426 = MULHI_INT_cm
  { 427,	21,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4a40ULL, nullptr, nullptr, OperandInfo56 },  // Inst #427 = MULHI_INT_cm24
  { 428,	21,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #428 = MULHI_INT_eg
  { 429,	21,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #429 = MULHI_INT_r600
  { 430,	21,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #430 = MULHI_UINT24_eg
  { 431,	21,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4a40ULL, nullptr, nullptr, OperandInfo56 },  // Inst #431 = MULHI_UINT_cm
  { 432,	21,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4a40ULL, nullptr, nullptr, OperandInfo56 },  // Inst #432 = MULHI_UINT_cm24
  { 433,	21,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #433 = MULHI_UINT_eg
  { 434,	21,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #434 = MULHI_UINT_r600
  { 435,	21,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4a40ULL, nullptr, nullptr, OperandInfo56 },  // Inst #435 = MULLO_INT_cm
  { 436,	21,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #436 = MULLO_INT_eg
  { 437,	21,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #437 = MULLO_INT_r600
  { 438,	21,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4a40ULL, nullptr, nullptr, OperandInfo56 },  // Inst #438 = MULLO_UINT_cm
  { 439,	21,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #439 = MULLO_UINT_eg
  { 440,	21,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #440 = MULLO_UINT_r600
  { 441,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #441 = MUL_IEEE
  { 442,	21,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #442 = MUL_INT24_cm
  { 443,	19,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #443 = MUL_LIT_eg
  { 444,	19,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4220ULL, nullptr, nullptr, OperandInfo58 },  // Inst #444 = MUL_LIT_r600
  { 445,	21,	1,	0,	2,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #445 = MUL_UINT24_eg
  { 446,	14,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #446 = NOT_INT
  { 447,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #447 = OR_INT
  { 448,	0,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, nullptr },  // Inst #448 = PAD
  { 449,	2,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo8 },  // Inst #449 = POP_EG
  { 450,	2,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo8 },  // Inst #450 = POP_R600
  { 451,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #451 = PRED_SETE
  { 452,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #452 = PRED_SETE_INT
  { 453,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #453 = PRED_SETGE
  { 454,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #454 = PRED_SETGE_INT
  { 455,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #455 = PRED_SETGT
  { 456,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #456 = PRED_SETGT_INT
  { 457,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #457 = PRED_SETNE
  { 458,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #458 = PRED_SETNE_INT
  { 459,	7,	0,	0,	1,	0|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo60 },  // Inst #459 = R600_ExportBuf
  { 460,	9,	0,	0,	1,	0|(1ULL<<MCID::NotDuplicable)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo61 },  // Inst #460 = R600_ExportSwz
  { 461,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #461 = RAT_ATOMIC_ADD_NORET
  { 462,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #462 = RAT_ATOMIC_ADD_RTN
  { 463,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #463 = RAT_ATOMIC_AND_NORET
  { 464,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #464 = RAT_ATOMIC_AND_RTN
  { 465,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #465 = RAT_ATOMIC_CMPXCHG_INT_NORET
  { 466,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #466 = RAT_ATOMIC_CMPXCHG_INT_RTN
  { 467,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #467 = RAT_ATOMIC_DEC_UINT_NORET
  { 468,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #468 = RAT_ATOMIC_DEC_UINT_RTN
  { 469,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #469 = RAT_ATOMIC_INC_UINT_NORET
  { 470,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #470 = RAT_ATOMIC_INC_UINT_RTN
  { 471,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #471 = RAT_ATOMIC_MAX_INT_NORET
  { 472,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #472 = RAT_ATOMIC_MAX_INT_RTN
  { 473,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #473 = RAT_ATOMIC_MAX_UINT_NORET
  { 474,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #474 = RAT_ATOMIC_MAX_UINT_RTN
  { 475,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #475 = RAT_ATOMIC_MIN_INT_NORET
  { 476,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #476 = RAT_ATOMIC_MIN_INT_RTN
  { 477,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #477 = RAT_ATOMIC_MIN_UINT_NORET
  { 478,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #478 = RAT_ATOMIC_MIN_UINT_RTN
  { 479,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #479 = RAT_ATOMIC_OR_NORET
  { 480,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #480 = RAT_ATOMIC_OR_RTN
  { 481,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #481 = RAT_ATOMIC_RSUB_NORET
  { 482,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #482 = RAT_ATOMIC_RSUB_RTN
  { 483,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #483 = RAT_ATOMIC_SUB_NORET
  { 484,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #484 = RAT_ATOMIC_SUB_RTN
  { 485,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #485 = RAT_ATOMIC_XCHG_INT_NORET
  { 486,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #486 = RAT_ATOMIC_XCHG_INT_RTN
  { 487,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #487 = RAT_ATOMIC_XOR_NORET
  { 488,	3,	1,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo70 },  // Inst #488 = RAT_ATOMIC_XOR_RTN
  { 489,	2,	0,	0,	1,	0|(1ULL<<MCID::MayStore), 0x20000ULL, nullptr, nullptr, OperandInfo71 },  // Inst #489 = RAT_MSKOR
  { 490,	2,	0,	0,	1,	0|(1ULL<<MCID::MayStore), 0x20000ULL, nullptr, nullptr, OperandInfo71 },  // Inst #490 = RAT_STORE_DWORD128
  { 491,	2,	0,	0,	1,	0|(1ULL<<MCID::MayStore), 0x20000ULL, nullptr, nullptr, OperandInfo72 },  // Inst #491 = RAT_STORE_DWORD32
  { 492,	2,	0,	0,	1,	0|(1ULL<<MCID::MayStore), 0x20000ULL, nullptr, nullptr, OperandInfo73 },  // Inst #492 = RAT_STORE_DWORD64
  { 493,	4,	0,	0,	1,	0|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo74 },  // Inst #493 = RAT_STORE_TYPED_cm
  { 494,	4,	0,	0,	1,	0|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::UnmodeledSideEffects), 0x20000ULL, nullptr, nullptr, OperandInfo74 },  // Inst #494 = RAT_STORE_TYPED_eg
  { 495,	3,	0,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UsesCustomInserter), 0x20000ULL, nullptr, nullptr, OperandInfo75 },  // Inst #495 = RAT_WRITE_CACHELESS_128_eg
  { 496,	3,	0,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UsesCustomInserter), 0x20000ULL, nullptr, nullptr, OperandInfo76 },  // Inst #496 = RAT_WRITE_CACHELESS_32_eg
  { 497,	3,	0,	0,	1,	0|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UsesCustomInserter), 0x20000ULL, nullptr, nullptr, OperandInfo77 },  // Inst #497 = RAT_WRITE_CACHELESS_64_eg
  { 498,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4640ULL, nullptr, nullptr, OperandInfo57 },  // Inst #498 = RECIPSQRT_CLAMPED_cm
  { 499,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #499 = RECIPSQRT_CLAMPED_eg
  { 500,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #500 = RECIPSQRT_CLAMPED_r600
  { 501,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4640ULL, nullptr, nullptr, OperandInfo57 },  // Inst #501 = RECIPSQRT_IEEE_cm
  { 502,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #502 = RECIPSQRT_IEEE_eg
  { 503,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #503 = RECIPSQRT_IEEE_r600
  { 504,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4640ULL, nullptr, nullptr, OperandInfo57 },  // Inst #504 = RECIP_CLAMPED_cm
  { 505,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #505 = RECIP_CLAMPED_eg
  { 506,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #506 = RECIP_CLAMPED_r600
  { 507,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4640ULL, nullptr, nullptr, OperandInfo57 },  // Inst #507 = RECIP_IEEE_cm
  { 508,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #508 = RECIP_IEEE_eg
  { 509,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #509 = RECIP_IEEE_r600
  { 510,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #510 = RECIP_UINT_eg
  { 511,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #511 = RECIP_UINT_r600
  { 512,	14,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #512 = RNDNE
  { 513,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #513 = SETE
  { 514,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #514 = SETE_DX10
  { 515,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #515 = SETE_INT
  { 516,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #516 = SETGE_DX10
  { 517,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #517 = SETGE_INT
  { 518,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #518 = SETGE_UINT
  { 519,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #519 = SETGT_DX10
  { 520,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #520 = SETGT_INT
  { 521,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #521 = SETGT_UINT
  { 522,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #522 = SETNE_DX10
  { 523,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #523 = SETNE_INT
  { 524,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #524 = SGE
  { 525,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #525 = SGT
  { 526,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4650ULL, nullptr, nullptr, OperandInfo57 },  // Inst #526 = SIN_cm
  { 527,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4610ULL, nullptr, nullptr, OperandInfo57 },  // Inst #527 = SIN_eg
  { 528,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4610ULL, nullptr, nullptr, OperandInfo57 },  // Inst #528 = SIN_r600
  { 529,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4610ULL, nullptr, nullptr, OperandInfo57 },  // Inst #529 = SIN_r700
  { 530,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #530 = SNE
  { 531,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #531 = SUBB_UINT
  { 532,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #532 = SUB_INT
  { 533,	19,	1,	0,	1,	0, 0x2000ULL, nullptr, nullptr, OperandInfo78 },  // Inst #533 = TEX_GET_GRADIENTS_H
  { 534,	19,	1,	0,	1,	0, 0x2000ULL, nullptr, nullptr, OperandInfo78 },  // Inst #534 = TEX_GET_GRADIENTS_V
  { 535,	19,	1,	0,	1,	0, 0x2000ULL, nullptr, nullptr, OperandInfo78 },  // Inst #535 = TEX_GET_TEXTURE_RESINFO
  { 536,	19,	1,	0,	1,	0, 0x2000ULL, nullptr, nullptr, OperandInfo78 },  // Inst #536 = TEX_LD
  { 537,	19,	1,	0,	1,	0, 0x2000ULL, nullptr, nullptr, OperandInfo78 },  // Inst #537 = TEX_LDPTR
  { 538,	19,	1,	0,	1,	0, 0x2000ULL, nullptr, nullptr, OperandInfo78 },  // Inst #538 = TEX_SAMPLE
  { 539,	19,	1,	0,	1,	0, 0x2000ULL, nullptr, nullptr, OperandInfo78 },  // Inst #539 = TEX_SAMPLE_C
  { 540,	19,	1,	0,	1,	0, 0x2000ULL, nullptr, nullptr, OperandInfo78 },  // Inst #540 = TEX_SAMPLE_C_G
  { 541,	19,	1,	0,	1,	0, 0x2000ULL, nullptr, nullptr, OperandInfo78 },  // Inst #541 = TEX_SAMPLE_C_L
  { 542,	19,	1,	0,	1,	0, 0x2000ULL, nullptr, nullptr, OperandInfo78 },  // Inst #542 = TEX_SAMPLE_C_LB
  { 543,	19,	1,	0,	1,	0, 0x2000ULL, nullptr, nullptr, OperandInfo78 },  // Inst #543 = TEX_SAMPLE_G
  { 544,	19,	1,	0,	1,	0, 0x2000ULL, nullptr, nullptr, OperandInfo78 },  // Inst #544 = TEX_SAMPLE_L
  { 545,	19,	1,	0,	1,	0, 0x2000ULL, nullptr, nullptr, OperandInfo78 },  // Inst #545 = TEX_SAMPLE_LB
  { 546,	19,	1,	0,	1,	0, 0x2000ULL, nullptr, nullptr, OperandInfo78 },  // Inst #546 = TEX_SET_GRADIENTS_H
  { 547,	19,	1,	0,	1,	0, 0x2000ULL, nullptr, nullptr, OperandInfo78 },  // Inst #547 = TEX_SET_GRADIENTS_V
  { 548,	4,	1,	0,	1,	0, 0x1000ULL, nullptr, nullptr, OperandInfo79 },  // Inst #548 = TEX_VTX_CONSTBUF
  { 549,	4,	1,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x1000ULL, nullptr, nullptr, OperandInfo79 },  // Inst #549 = TEX_VTX_TEXBUF
  { 550,	14,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #550 = TRUNC
  { 551,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #551 = UINT_TO_FLT_eg
  { 552,	14,	1,	0,	4,	0|(1ULL<<MCID::Predicable), 0x4600ULL, nullptr, nullptr, OperandInfo57 },  // Inst #552 = UINT_TO_FLT_r600
  { 553,	4,	1,	0,	1,	0|(1ULL<<MCID::MayLoad), 0x1000ULL, nullptr, nullptr, OperandInfo79 },  // Inst #553 = VTX_READ_128_cm
  { 554,	4,	1,	0,	1,	0|(1ULL<<MCID::MayLoad), 0x1000ULL, nullptr, nullptr, OperandInfo79 },  // Inst #554 = VTX_READ_128_eg
  { 555,	4,	1,	0,	1,	0|(1ULL<<MCID::MayLoad), 0x1000ULL, nullptr, nullptr, OperandInfo80 },  // Inst #555 = VTX_READ_16_cm
  { 556,	4,	1,	0,	1,	0|(1ULL<<MCID::MayLoad), 0x1000ULL, nullptr, nullptr, OperandInfo80 },  // Inst #556 = VTX_READ_16_eg
  { 557,	4,	1,	0,	1,	0|(1ULL<<MCID::MayLoad), 0x1000ULL, nullptr, nullptr, OperandInfo81 },  // Inst #557 = VTX_READ_32_cm
  { 558,	4,	1,	0,	1,	0|(1ULL<<MCID::MayLoad), 0x1000ULL, nullptr, nullptr, OperandInfo81 },  // Inst #558 = VTX_READ_32_eg
  { 559,	4,	1,	0,	1,	0|(1ULL<<MCID::MayLoad), 0x1000ULL, nullptr, nullptr, OperandInfo82 },  // Inst #559 = VTX_READ_64_cm
  { 560,	4,	1,	0,	1,	0|(1ULL<<MCID::MayLoad), 0x1000ULL, nullptr, nullptr, OperandInfo82 },  // Inst #560 = VTX_READ_64_eg
  { 561,	4,	1,	0,	1,	0|(1ULL<<MCID::MayLoad), 0x1000ULL, nullptr, nullptr, OperandInfo80 },  // Inst #561 = VTX_READ_8_cm
  { 562,	4,	1,	0,	1,	0|(1ULL<<MCID::MayLoad), 0x1000ULL, nullptr, nullptr, OperandInfo80 },  // Inst #562 = VTX_READ_8_eg
  { 563,	1,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo3 },  // Inst #563 = WHILE_LOOP_EG
  { 564,	1,	0,	0,	1,	0|(1ULL<<MCID::UnmodeledSideEffects), 0x0ULL, nullptr, nullptr, OperandInfo3 },  // Inst #564 = WHILE_LOOP_R600
  { 565,	21,	1,	0,	3,	0|(1ULL<<MCID::Predicable), 0x4a00ULL, nullptr, nullptr, OperandInfo56 },  // Inst #565 = XOR_INT
};

extern const char R600InstrNameData[] = {
  /* 0 */ 'C', 'F', '_', 'T', 'C', '_', 'R', '6', '0', '0', 0,
  /* 11 */ 'C', 'F', '_', 'V', 'C', '_', 'R', '6', '0', '0', 0,
  /* 22 */ 'C', 'F', '_', 'E', 'N', 'D', '_', 'R', '6', '0', '0', 0,
  /* 34 */ 'C', 'F', '_', 'E', 'L', 'S', 'E', '_', 'R', '6', '0', '0', 0,
  /* 47 */ 'C', 'F', '_', 'P', 'U', 'S', 'H', '_', 'E', 'L', 'S', 'E', '_', 'R', '6', '0', '0', 0,
  /* 65 */ 'C', 'F', '_', 'C', 'O', 'N', 'T', 'I', 'N', 'U', 'E', '_', 'R', '6', '0', '0', 0,
  /* 82 */ 'F', 'N', 'E', 'G', '_', 'R', '6', '0', '0', 0,
  /* 92 */ 'L', 'O', 'O', 'P', '_', 'B', 'R', 'E', 'A', 'K', '_', 'R', '6', '0', '0', 0,
  /* 108 */ 'C', 'F', '_', 'J', 'U', 'M', 'P', '_', 'R', '6', '0', '0', 0,
  /* 121 */ 'E', 'N', 'D', '_', 'L', 'O', 'O', 'P', '_', 'R', '6', '0', '0', 0,
  /* 135 */ 'W', 'H', 'I', 'L', 'E', '_', 'L', 'O', 'O', 'P', '_', 'R', '6', '0', '0', 0,
  /* 151 */ 'P', 'O', 'P', '_', 'R', '6', '0', '0', 0,
  /* 160 */ 'F', 'A', 'B', 'S', '_', 'R', '6', '0', '0', 0,
  /* 170 */ 'C', 'F', '_', 'C', 'A', 'L', 'L', '_', 'F', 'S', '_', 'R', '6', '0', '0', 0,
  /* 186 */ 'D', 'O', 'T', '4', '_', 'r', '6', '0', '0', 0,
  /* 196 */ 'M', 'U', 'L', 'A', 'D', 'D', '_', 'r', '6', '0', '0', 0,
  /* 208 */ 'L', 'O', 'G', '_', 'C', 'L', 'A', 'M', 'P', 'E', 'D', '_', 'r', '6', '0', '0', 0,
  /* 225 */ 'R', 'E', 'C', 'I', 'P', '_', 'C', 'L', 'A', 'M', 'P', 'E', 'D', '_', 'r', '6', '0', '0', 0,
  /* 244 */ 'R', 'E', 'C', 'I', 'P', 'S', 'Q', 'R', 'T', '_', 'C', 'L', 'A', 'M', 'P', 'E', 'D', '_', 'r', '6', '0', '0', 0,
  /* 267 */ 'C', 'N', 'D', 'E', '_', 'r', '6', '0', '0', 0,
  /* 277 */ 'M', 'U', 'L', 'A', 'D', 'D', '_', 'I', 'E', 'E', 'E', '_', 'r', '6', '0', '0', 0,
  /* 294 */ 'L', 'O', 'G', '_', 'I', 'E', 'E', 'E', '_', 'r', '6', '0', '0', 0,
  /* 308 */ 'R', 'E', 'C', 'I', 'P', '_', 'I', 'E', 'E', 'E', '_', 'r', '6', '0', '0', 0,
  /* 324 */ 'E', 'X', 'P', '_', 'I', 'E', 'E', 'E', '_', 'r', '6', '0', '0', 0,
  /* 338 */ 'R', 'E', 'C', 'I', 'P', 'S', 'Q', 'R', 'T', '_', 'I', 'E', 'E', 'E', '_', 'r', '6', '0', '0', 0,
  /* 358 */ 'C', 'N', 'D', 'G', 'E', '_', 'r', '6', '0', '0', 0,
  /* 369 */ 'L', 'S', 'H', 'L', '_', 'r', '6', '0', '0', 0,
  /* 379 */ 'S', 'I', 'N', '_', 'r', '6', '0', '0', 0,
  /* 388 */ 'A', 'S', 'H', 'R', '_', 'r', '6', '0', '0', 0,
  /* 398 */ 'L', 'S', 'H', 'R', '_', 'r', '6', '0', '0', 0,
  /* 408 */ 'C', 'O', 'S', '_', 'r', '6', '0', '0', 0,
  /* 417 */ 'C', 'N', 'D', 'G', 'T', '_', 'r', '6', '0', '0', 0,
  /* 428 */ 'M', 'U', 'L', '_', 'L', 'I', 'T', '_', 'r', '6', '0', '0', 0,
  /* 441 */ 'U', 'I', 'N', 'T', '_', 'T', 'O', '_', 'F', 'L', 'T', '_', 'r', '6', '0', '0', 0,
  /* 458 */ 'M', 'U', 'L', 'H', 'I', '_', 'U', 'I', 'N', 'T', '_', 'r', '6', '0', '0', 0,
  /* 474 */ 'M', 'U', 'L', 'L', 'O', '_', 'U', 'I', 'N', 'T', '_', 'r', '6', '0', '0', 0,
  /* 490 */ 'F', 'L', 'T', '_', 'T', 'O', '_', 'U', 'I', 'N', 'T', '_', 'r', '6', '0', '0', 0,
  /* 507 */ 'R', 'E', 'C', 'I', 'P', '_', 'U', 'I', 'N', 'T', '_', 'r', '6', '0', '0', 0,
  /* 523 */ 'M', 'U', 'L', 'H', 'I', '_', 'I', 'N', 'T', '_', 'r', '6', '0', '0', 0,
  /* 538 */ 'M', 'U', 'L', 'L', 'O', '_', 'I', 'N', 'T', '_', 'r', '6', '0', '0', 0,
  /* 553 */ 'F', 'L', 'T', '_', 'T', 'O', '_', 'I', 'N', 'T', '_', 'r', '6', '0', '0', 0,
  /* 569 */ 'S', 'I', 'N', '_', 'r', '7', '0', '0', 0,
  /* 578 */ 'C', 'O', 'S', '_', 'r', '7', '0', '0', 0,
  /* 587 */ 'G', '_', 'F', 'L', 'O', 'G', '1', '0', 0,
  /* 596 */ 'S', 'E', 'T', 'G', 'E', '_', 'D', 'X', '1', '0', 0,
  /* 607 */ 'S', 'E', 'T', 'N', 'E', '_', 'D', 'X', '1', '0', 0,
  /* 618 */ 'S', 'E', 'T', 'E', '_', 'D', 'X', '1', '0', 0,
  /* 628 */ 'M', 'I', 'N', '_', 'D', 'X', '1', '0', 0,
  /* 637 */ 'S', 'E', 'T', 'G', 'T', '_', 'D', 'X', '1', '0', 0,
  /* 648 */ 'M', 'A', 'X', '_', 'D', 'X', '1', '0', 0,
  /* 657 */ 'I', 'N', 'T', 'E', 'R', 'P', '_', 'L', 'O', 'A', 'D', '_', 'P', '0', 0,
  /* 672 */ 'R', 'A', 'T', '_', 'S', 'T', 'O', 'R', 'E', '_', 'D', 'W', 'O', 'R', 'D', '3', '2', 0,
  /* 690 */ 'M', 'O', 'V', '_', 'I', 'M', 'M', '_', 'F', '3', '2', 0,
  /* 702 */ 'M', 'O', 'V', '_', 'I', 'M', 'M', '_', 'I', '3', '2', 0,
  /* 714 */ 'F', 'L', 'T', '1', '6', '_', 'T', 'O', '_', 'F', 'L', 'T', '3', '2', 0,
  /* 729 */ 'C', 'O', 'N', 'T', 'I', 'N', 'U', 'E', 'C', '_', 'f', '3', '2', 0,
  /* 743 */ 'I', 'F', 'C', '_', 'f', '3', '2', 0,
  /* 751 */ 'B', 'R', 'E', 'A', 'K', 'C', '_', 'f', '3', '2', 0,
  /* 762 */ 'B', 'R', 'A', 'N', 'C', 'H', '_', 'C', 'O', 'N', 'D', '_', 'f', '3', '2', 0,
  /* 778 */ 'C', 'O', 'N', 'T', 'I', 'N', 'U', 'E', '_', 'L', 'O', 'G', 'I', 'C', 'A', 'L', 'Z', '_', 'f', '3', '2', 0,
  /* 800 */ 'I', 'F', '_', 'L', 'O', 'G', 'I', 'C', 'A', 'L', 'Z', '_', 'f', '3', '2', 0,
  /* 816 */ 'B', 'R', 'E', 'A', 'K', '_', 'L', 'O', 'G', 'I', 'C', 'A', 'L', 'Z', '_', 'f', '3', '2', 0,
  /* 835 */ 'C', 'O', 'N', 'T', 'I', 'N', 'U', 'E', '_', 'L', 'O', 'G', 'I', 'C', 'A', 'L', 'N', 'Z', '_', 'f', '3', '2', 0,
  /* 858 */ 'I', 'F', '_', 'L', 'O', 'G', 'I', 'C', 'A', 'L', 'N', 'Z', '_', 'f', '3', '2', 0,
  /* 875 */ 'B', 'R', 'E', 'A', 'K', '_', 'L', 'O', 'G', 'I', 'C', 'A', 'L', 'N', 'Z', '_', 'f', '3', '2', 0,
  /* 895 */ 'C', 'O', 'N', 'T', 'I', 'N', 'U', 'E', 'C', '_', 'i', '3', '2', 0,
  /* 909 */ 'I', 'F', 'C', '_', 'i', '3', '2', 0,
  /* 917 */ 'B', 'R', 'E', 'A', 'K', 'C', '_', 'i', '3', '2', 0,
  /* 928 */ 'B', 'R', 'A', 'N', 'C', 'H', '_', 'C', 'O', 'N', 'D', '_', 'i', '3', '2', 0,
  /* 944 */ 'C', 'O', 'N', 'T', 'I', 'N', 'U', 'E', '_', 'L', 'O', 'G', 'I', 'C', 'A', 'L', 'Z', '_', 'i', '3', '2', 0,
  /* 966 */ 'I', 'F', '_', 'L', 'O', 'G', 'I', 'C', 'A', 'L', 'Z', '_', 'i', '3', '2', 0,
  /* 982 */ 'B', 'R', 'E', 'A', 'K', '_', 'L', 'O', 'G', 'I', 'C', 'A', 'L', 'Z', '_', 'i', '3', '2', 0,
  /* 1001 */ 'C', 'O', 'N', 'T', 'I', 'N', 'U', 'E', '_', 'L', 'O', 'G', 'I', 'C', 'A', 'L', 'N', 'Z', '_', 'i', '3', '2', 0,
  /* 1024 */ 'I', 'F', '_', 'L', 'O', 'G', 'I', 'C', 'A', 'L', 'N', 'Z', '_', 'i', '3', '2', 0,
  /* 1041 */ 'B', 'R', 'E', 'A', 'K', '_', 'L', 'O', 'G', 'I', 'C', 'A', 'L', 'N', 'Z', '_', 'i', '3', '2', 0,
  /* 1061 */ 'G', '_', 'F', 'L', 'O', 'G', '2', 0,
  /* 1069 */ 'G', '_', 'F', 'E', 'X', 'P', '2', 0,
  /* 1077 */ 'R', '6', '0', '0', '_', 'E', 'X', 'T', 'R', 'A', 'C', 'T', '_', 'E', 'L', 'T', '_', 'V', '2', 0,
  /* 1097 */ 'R', '6', '0', '0', '_', 'I', 'N', 'S', 'E', 'R', 'T', '_', 'E', 'L', 'T', '_', 'V', '2', 0,
  /* 1116 */ 'M', 'U', 'L', 'H', 'I', '_', 'U', 'I', 'N', 'T', '_', 'c', 'm', '2', '4', 0,
  /* 1132 */ 'M', 'U', 'L', 'H', 'I', '_', 'I', 'N', 'T', '_', 'c', 'm', '2', '4', 0,
  /* 1147 */ 'R', 'A', 'T', '_', 'S', 'T', 'O', 'R', 'E', '_', 'D', 'W', 'O', 'R', 'D', '6', '4', 0,
  /* 1165 */ 'R', '6', '0', '0', '_', 'E', 'X', 'T', 'R', 'A', 'C', 'T', '_', 'E', 'L', 'T', '_', 'V', '4', 0,
  /* 1185 */ 'R', '6', '0', '0', '_', 'I', 'N', 'S', 'E', 'R', 'T', '_', 'E', 'L', 'T', '_', 'V', '4', 0,
  /* 1204 */ 'D', 'O', 'T', '_', '4', 0,
  /* 1210 */ 'F', 'L', 'T', '3', '2', '_', 'T', 'O', '_', 'F', 'L', 'T', '1', '6', 0,
  /* 1225 */ 'R', 'A', 'T', '_', 'S', 'T', 'O', 'R', 'E', '_', 'D', 'W', 'O', 'R', 'D', '1', '2', '8', 0,
  /* 1244 */ 'G', '_', 'F', 'M', 'A', 0,
  /* 1250 */ 'G', '_', 'S', 'T', 'R', 'I', 'C', 'T', '_', 'F', 'M', 'A', 0,
  /* 1263 */ 'T', 'E', 'X', '_', 'S', 'A', 'M', 'P', 'L', 'E', '_', 'C', '_', 'L', 'B', 0,
  /* 1279 */ 'T', 'E', 'X', '_', 'S', 'A', 'M', 'P', 'L', 'E', '_', 'L', 'B', 0,
  /* 1293 */ 'G', '_', 'F', 'S', 'U', 'B', 0,
  /* 1300 */ 'G', '_', 'S', 'T', 'R', 'I', 'C', 'T', '_', 'F', 'S', 'U', 'B', 0,
  /* 1314 */ 'G', '_', 'A', 'T', 'O', 'M', 'I', 'C', 'R', 'M', 'W', '_', 'F', 'S', 'U', 'B', 0,
  /* 1331 */ 'G', '_', 'S', 'U', 'B', 0,
  /* 1337 */ 'L', 'D', 'S', '_', 'S', 'U', 'B', 0,
  /* 1345 */ 'G', '_', 'A', 'T', 'O', 'M', 'I', 'C', 'R', 'M', 'W', '_', 'S', 'U', 'B', 0,
  /* 1361 */ 'G', '_', 'I', 'N', 'T', 'R', 'I', 'N', 'S', 'I', 'C', 0,
  /* 1373 */ 'E', 'N', 'D', 'F', 'U', 'N', 'C', 0,
  /* 1381 */ 'G', '_', 'F', 'P', 'T', 'R', 'U', 'N', 'C', 0,
  /* 1391 */ 'G', '_', 'I', 'N', 'T', 'R', 'I', 'N', 'S', 'I', 'C', '_', 'T', 'R', 'U', 'N', 'C', 0,
  /* 1409 */ 'G', '_', 'T', 'R', 'U', 'N', 'C', 0,
  /* 1417 */ 'G', '_', 'B', 'U', 'I', 'L', 'D', '_', 'V', 'E', 'C', 'T', 'O', 'R', '_', 'T', 'R', 'U', 'N', 'C', 0,
  /* 1438 */ 'G', '_', 'D', 'Y', 'N', '_', 'S', 'T', 'A', 'C', 'K', 'A', 'L', 'L', 'O', 'C', 0,
  /* 1455 */ 'T', 'E', 'X', '_', 'S', 'A', 'M', 'P', 'L', 'E', '_', 'C', 0,
  /* 1468 */ 'G', '_', 'F', 'M', 'A', 'D', 0,
  /* 1475 */ 'G', '_', 'I', 'N', 'D', 'E', 'X', 'E', 'D', '_', 'S', 'E', 'X', 'T', 'L', 'O', 'A', 'D', 0,
  /* 1494 */ 'G', '_', 'S', 'E', 'X', 'T', 'L', 'O', 'A', 'D', 0,
  /* 1505 */ 'G', '_', 'I', 'N', 'D', 'E', 'X', 'E', 'D', '_', 'Z', 'E', 'X', 'T', 'L', 'O', 'A', 'D', 0,
  /* 1524 */ 'G', '_', 'Z', 'E', 'X', 'T', 'L', 'O', 'A', 'D', 0,
  /* 1535 */ 'I', 'N', 'T', 'E', 'R', 'P', '_', 'V', 'E', 'C', '_', 'L', 'O', 'A', 'D', 0,
  /* 1551 */ 'G', '_', 'I', 'N', 'D', 'E', 'X', 'E', 'D', '_', 'L', 'O', 'A', 'D', 0,
  /* 1566 */ 'G', '_', 'L', 'O', 'A', 'D', 0,
  /* 1573 */ 'P', 'A', 'D', 0,
  /* 1577 */ 'G', '_', 'V', 'E', 'C', 'R', 'E', 'D', 'U', 'C', 'E', '_', 'F', 'A', 'D', 'D', 0,
  /* 1594 */ 'G', '_', 'F', 'A', 'D', 'D', 0,
  /* 1601 */ 'G', '_', 'V', 'E', 'C', 'R', 'E', 'D', 'U', 'C', 'E', '_', 'S', 'E', 'Q', '_', 'F', 'A', 'D', 'D', 0,
  /* 1622 */ 'G', '_', 'S', 'T', 'R', 'I', 'C', 'T', '_', 'F', 'A', 'D', 'D', 0,
  /* 1636 */ 'G', '_', 'A', 'T', 'O', 'M', 'I', 'C', 'R', 'M', 'W', '_', 'F', 'A', 'D', 'D', 0,
  /* 1653 */ 'G', '_', 'V', 'E', 'C', 'R', 'E', 'D', 'U', 'C', 'E', '_', 'A', 'D', 'D', 0,
  /* 1669 */ 'G', '_', 'A', 'D', 'D', 0,
  /* 1675 */ 'G', '_', 'P', 'T', 'R', '_', 'A', 'D', 'D', 0,
  /* 1685 */ 'L', 'D', 'S', '_', 'A', 'D', 'D', 0,
  /* 1693 */ 'G', '_', 'A', 'T', 'O', 'M', 'I', 'C', 'R', 'M', 'W', '_', 'A', 'D', 'D', 0,
  /* 1709 */ 'T', 'E', 'X', '_', 'L', 'D', 0,
  /* 1716 */ 'G', '_', 'A', 'T', 'O', 'M', 'I', 'C', 'R', 'M', 'W', '_', 'N', 'A', 'N', 'D', 0,
  /* 1733 */ 'G', '_', 'V', 'E', 'C', 'R', 'E', 'D', 'U', 'C', 'E', '_', 'A', 'N', 'D', 0,
  /* 1749 */ 'G', '_', 'A', 'N', 'D', 0,
  /* 1755 */ 'L', 'D', 'S', '_', 'A', 'N', 'D', 0,
  /* 1763 */ 'G', '_', 'A', 'T', 'O', 'M', 'I', 'C', 'R', 'M', 'W', '_', 'A', 'N', 'D', 0,
  /* 1779 */ 'L', 'I', 'F', 'E', 'T', 'I', 'M', 'E', '_', 'E', 'N', 'D', 0,
  /* 1792 */ 'G', '_', 'B', 'R', 'C', 'O', 'N', 'D', 0,
  /* 1801 */ 'J', 'U', 'M', 'P', '_', 'C', 'O', 'N', 'D', 0,
  /* 1811 */ 'G', '_', 'I', 'N', 'T', 'R', 'I', 'N', 'S', 'I', 'C', '_', 'R', 'O', 'U', 'N', 'D', 0,
  /* 1829 */ 'L', 'O', 'A', 'D', '_', 'S', 'T', 'A', 'C', 'K', '_', 'G', 'U', 'A', 'R', 'D', 0,
  /* 1846 */ 'T', 'X', 'D', 0,
  /* 1850 */ 'G', '_', 'S', 'S', 'U', 'B', 'E', 0,
  /* 1858 */ 'G', '_', 'U', 'S', 'U', 'B', 'E', 0,
  /* 1866 */ 'G', '_', 'F', 'E', 'N', 'C', 'E', 0,
  /* 1874 */ 'R', 'E', 'G', '_', 'S', 'E', 'Q', 'U', 'E', 'N', 'C', 'E', 0,
  /* 1887 */ 'G', '_', 'S', 'A', 'D', 'D', 'E', 0,
  /* 1895 */ 'G', '_', 'U', 'A', 'D', 'D', 'E', 0,
  /* 1903 */ 'M', 'U', 'L', '_', 'I', 'E', 'E', 'E', 0,
  /* 1912 */ 'G', '_', 'F', 'M', 'I', 'N', 'N', 'U', 'M', '_', 'I', 'E', 'E', 'E', 0,
  /* 1927 */ 'G', '_', 'F', 'M', 'A', 'X', 'N', 'U', 'M', '_', 'I', 'E', 'E', 'E', 0,
  /* 1942 */ 'S', 'G', 'E', 0,
  /* 1946 */ 'P', 'R', 'E', 'D', '_', 'S', 'E', 'T', 'G', 'E', 0,
  /* 1957 */ 'G', '_', 'J', 'U', 'M', 'P', '_', 'T', 'A', 'B', 'L', 'E', 0,
  /* 1970 */ 'B', 'U', 'N', 'D', 'L', 'E', 0,
  /* 1977 */ 'T', 'E', 'X', '_', 'S', 'A', 'M', 'P', 'L', 'E', 0,
  /* 1988 */ 'R', 'N', 'D', 'N', 'E', 0,
  /* 1994 */ 'S', 'N', 'E', 0,
  /* 1998 */ 'P', 'R', 'E', 'D', '_', 'S', 'E', 'T', 'N', 'E', 0,
  /* 2009 */ 'L', 'O', 'C', 'A', 'L', '_', 'E', 'S', 'C', 'A', 'P', 'E', 0,
  /* 2022 */ 'C', 'F', '_', 'A', 'L', 'U', '_', 'P', 'U', 'S', 'H', '_', 'B', 'E', 'F', 'O', 'R', 'E', 0,
  /* 2041 */ 'G', '_', 'I', 'N', 'D', 'E', 'X', 'E', 'D', '_', 'S', 'T', 'O', 'R', 'E', 0,
  /* 2057 */ 'G', '_', 'S', 'T', 'O', 'R', 'E', 0,
  /* 2065 */ 'E', 'L', 'S', 'E', 0,
  /* 2070 */ 'G', '_', 'B', 'I', 'T', 'R', 'E', 'V', 'E', 'R', 'S', 'E', 0,
  /* 2083 */ 'F', 'E', 'T', 'C', 'H', '_', 'C', 'L', 'A', 'U', 'S', 'E', 0,
  /* 2096 */ 'A', 'L', 'U', '_', 'C', 'L', 'A', 'U', 'S', 'E', 0,
  /* 2107 */ 'P', 'R', 'E', 'D', '_', 'S', 'E', 'T', 'E', 0,
  /* 2117 */ 'L', 'D', 'S', '_', 'B', 'Y', 'T', 'E', '_', 'W', 'R', 'I', 'T', 'E', 0,
  /* 2132 */ 'M', 'A', 'S', 'K', '_', 'W', 'R', 'I', 'T', 'E', 0,
  /* 2143 */ 'L', 'D', 'S', '_', 'W', 'R', 'I', 'T', 'E', 0,
  /* 2153 */ 'L', 'D', 'S', '_', 'S', 'H', 'O', 'R', 'T', '_', 'W', 'R', 'I', 'T', 'E', 0,
  /* 2169 */ 'D', 'B', 'G', '_', 'V', 'A', 'L', 'U', 'E', 0,
  /* 2179 */ 'G', '_', 'G', 'L', 'O', 'B', 'A', 'L', '_', 'V', 'A', 'L', 'U', 'E', 0,
  /* 2194 */ 'C', 'F', '_', 'A', 'L', 'U', '_', 'C', 'O', 'N', 'T', 'I', 'N', 'U', 'E', 0,
  /* 2210 */ 'G', '_', 'M', 'E', 'M', 'M', 'O', 'V', 'E', 0,
  /* 2220 */ 'G', '_', 'F', 'R', 'E', 'E', 'Z', 'E', 0,
  /* 2229 */ 'G', '_', 'F', 'C', 'A', 'N', 'O', 'N', 'I', 'C', 'A', 'L', 'I', 'Z', 'E', 0,
  /* 2245 */ 'G', '_', 'C', 'T', 'L', 'Z', '_', 'Z', 'E', 'R', 'O', '_', 'U', 'N', 'D', 'E', 'F', 0,
  /* 2263 */ 'G', '_', 'C', 'T', 'T', 'Z', '_', 'Z', 'E', 'R', 'O', '_', 'U', 'N', 'D', 'E', 'F', 0,
  /* 2281 */ 'G', '_', 'I', 'M', 'P', 'L', 'I', 'C', 'I', 'T', '_', 'D', 'E', 'F', 0,
  /* 2296 */ 'D', 'B', 'G', '_', 'I', 'N', 'S', 'T', 'R', '_', 'R', 'E', 'F', 0,
  /* 2310 */ 'E', 'N', 'D', 'I', 'F', 0,
  /* 2316 */ 'T', 'E', 'X', '_', 'V', 'T', 'X', '_', 'C', 'O', 'N', 'S', 'T', 'B', 'U', 'F', 0,
  /* 2333 */ 'T', 'E', 'X', '_', 'V', 'T', 'X', '_', 'T', 'E', 'X', 'B', 'U', 'F', 0,
  /* 2348 */ 'G', '_', 'F', 'N', 'E', 'G', 0,
  /* 2355 */ 'E', 'X', 'T', 'R', 'A', 'C', 'T', '_', 'S', 'U', 'B', 'R', 'E', 'G', 0,
  /* 2370 */ 'I', 'N', 'S', 'E', 'R', 'T', '_', 'S', 'U', 'B', 'R', 'E', 'G', 0,
  /* 2384 */ 'G', '_', 'S', 'E', 'X', 'T', '_', 'I', 'N', 'R', 'E', 'G', 0,
  /* 2397 */ 'S', 'U', 'B', 'R', 'E', 'G', '_', 'T', 'O', '_', 'R', 'E', 'G', 0,
  /* 2411 */ 'C', 'F', '_', 'T', 'C', '_', 'E', 'G', 0,
  /* 2420 */ 'C', 'F', '_', 'V', 'C', '_', 'E', 'G', 0,
  /* 2429 */ 'C', 'F', '_', 'E', 'N', 'D', '_', 'E', 'G', 0,
  /* 2439 */ 'C', 'F', '_', 'E', 'L', 'S', 'E', '_', 'E', 'G', 0,
  /* 2450 */ 'C', 'F', '_', 'C', 'O', 'N', 'T', 'I', 'N', 'U', 'E', '_', 'E', 'G', 0,
  /* 2465 */ 'C', 'F', '_', 'P', 'U', 'S', 'H', '_', 'E', 'G', 0,
  /* 2476 */ 'L', 'O', 'O', 'P', '_', 'B', 'R', 'E', 'A', 'K', '_', 'E', 'G', 0,
  /* 2490 */ 'C', 'F', '_', 'J', 'U', 'M', 'P', '_', 'E', 'G', 0,
  /* 2501 */ 'E', 'N', 'D', '_', 'L', 'O', 'O', 'P', '_', 'E', 'G', 0,
  /* 2513 */ 'W', 'H', 'I', 'L', 'E', '_', 'L', 'O', 'O', 'P', '_', 'E', 'G', 0,
  /* 2527 */ 'P', 'O', 'P', '_', 'E', 'G', 0,
  /* 2534 */ 'C', 'F', '_', 'C', 'A', 'L', 'L', '_', 'F', 'S', '_', 'E', 'G', 0,
  /* 2548 */ 'G', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'C', 'M', 'P', 'X', 'C', 'H', 'G', 0,
  /* 2565 */ 'L', 'D', 'S', '_', 'W', 'R', 'X', 'C', 'H', 'G', 0,
  /* 2576 */ 'G', '_', 'A', 'T', 'O', 'M', 'I', 'C', 'R', 'M', 'W', '_', 'X', 'C', 'H', 'G', 0,
  /* 2593 */ 'G', '_', 'F', 'L', 'O', 'G', 0,
  /* 2600 */ 'G', '_', 'V', 'A', 'A', 'R', 'G', 0,
  /* 2608 */ 'P', 'R', 'E', 'A', 'L', 'L', 'O', 'C', 'A', 'T', 'E', 'D', '_', 'A', 'R', 'G', 0,
  /* 2625 */ 'T', 'E', 'X', '_', 'S', 'A', 'M', 'P', 'L', 'E', '_', 'C', '_', 'G', 0,
  /* 2640 */ 'T', 'E', 'X', '_', 'S', 'A', 'M', 'P', 'L', 'E', '_', 'G', 0,
  /* 2653 */ 'B', 'R', 'A', 'N', 'C', 'H', 0,
  /* 2660 */ 'E', 'N', 'D', 'S', 'W', 'I', 'T', 'C', 'H', 0,
  /* 2670 */ 'G', '_', 'S', 'M', 'U', 'L', 'H', 0,
  /* 2678 */ 'G', '_', 'U', 'M', 'U', 'L', 'H', 0,
  /* 2686 */ 'T', 'E', 'X', '_', 'G', 'E', 'T', '_', 'G', 'R', 'A', 'D', 'I', 'E', 'N', 'T', 'S', '_', 'H', 0,
  /* 2706 */ 'T', 'E', 'X', '_', 'S', 'E', 'T', '_', 'G', 'R', 'A', 'D', 'I', 'E', 'N', 'T', 'S', '_', 'H', 0,
  /* 2726 */ 'G', '_', 'P', 'H', 'I', 0,
  /* 2732 */ 'G', '_', 'F', 'P', 'T', 'O', 'S', 'I', 0,
  /* 2741 */ 'G', '_', 'F', 'P', 'T', 'O', 'U', 'I', 0,
  /* 2750 */ 'G', '_', 'F', 'P', 'O', 'W', 'I', 0,
  /* 2758 */ 'C', 'F', '_', 'A', 'L', 'U', '_', 'B', 'R', 'E', 'A', 'K', 0,
  /* 2771 */ 'G', '_', 'P', 'T', 'R', 'M', 'A', 'S', 'K', 0,
  /* 2781 */ 'G', 'C', '_', 'L', 'A', 'B', 'E', 'L', 0,
  /* 2790 */ 'D', 'B', 'G', '_', 'L', 'A', 'B', 'E', 'L', 0,
  /* 2800 */ 'E', 'H', '_', 'L', 'A', 'B', 'E', 'L', 0,
  /* 2809 */ 'A', 'N', 'N', 'O', 'T', 'A', 'T', 'I', 'O', 'N', '_', 'L', 'A', 'B', 'E', 'L', 0,
  /* 2826 */ 'I', 'C', 'A', 'L', 'L', '_', 'B', 'R', 'A', 'N', 'C', 'H', '_', 'F', 'U', 'N', 'N', 'E', 'L', 0,
  /* 2846 */ 'G', '_', 'F', 'S', 'H', 'L', 0,
  /* 2853 */ 'G', '_', 'S', 'H', 'L', 0,
  /* 2859 */ 'G', '_', 'F', 'C', 'E', 'I', 'L', 0,
  /* 2867 */ 'P', 'A', 'T', 'C', 'H', 'A', 'B', 'L', 'E', '_', 'T', 'A', 'I', 'L', '_', 'C', 'A', 'L', 'L', 0,
  /* 2887 */ 'P', 'A', 'T', 'C', 'H', 'A', 'B', 'L', 'E', '_', 'T', 'Y', 'P', 'E', 'D', '_', 'E', 'V', 'E', 'N', 'T', '_', 'C', 'A', 'L', 'L', 0,
  /* 2914 */ 'P', 'A', 'T', 'C', 'H', 'A', 'B', 'L', 'E', '_', 'E', 'V', 'E', 'N', 'T', '_', 'C', 'A', 'L', 'L', 0,
  /* 2935 */ 'F', 'E', 'N', 'T', 'R', 'Y', '_', 'C', 'A', 'L', 'L', 0,
  /* 2947 */ 'K', 'I', 'L', 'L', 0,
  /* 2952 */ 'G', '_', 'V', 'E', 'C', 'R', 'E', 'D', 'U', 'C', 'E', '_', 'F', 'M', 'U', 'L', 0,
  /* 2969 */ 'G', '_', 'F', 'M', 'U', 'L', 0,
  /* 2976 */ 'G', '_', 'V', 'E', 'C', 'R', 'E', 'D', 'U', 'C', 'E', '_', 'S', 'E', 'Q', '_', 'F', 'M', 'U', 'L', 0,
  /* 2997 */ 'G', '_', 'S', 'T', 'R', 'I', 'C', 'T', '_', 'F', 'M', 'U', 'L', 0,
  /* 3011 */ 'G', '_', 'V', 'E', 'C', 'R', 'E', 'D', 'U', 'C', 'E', '_', 'M', 'U', 'L', 0,
  /* 3027 */ 'G', '_', 'M', 'U', 'L', 0,
  /* 3033 */ 'T', 'E', 'X', '_', 'S', 'A', 'M', 'P', 'L', 'E', '_', 'C', '_', 'L', 0,
  /* 3048 */ 'T', 'E', 'X', '_', 'S', 'A', 'M', 'P', 'L', 'E', '_', 'L', 0,
  /* 3061 */ 'C', 'F', '_', 'E', 'N', 'D', '_', 'C', 'M', 0,
  /* 3071 */ 'G', '_', 'F', 'R', 'E', 'M', 0,
  /* 3078 */ 'G', '_', 'S', 'T', 'R', 'I', 'C', 'T', '_', 'F', 'R', 'E', 'M', 0,
  /* 3092 */ 'G', '_', 'S', 'R', 'E', 'M', 0,
  /* 3099 */ 'G', '_', 'U', 'R', 'E', 'M', 0,
  /* 3106 */ 'I', 'N', 'L', 'I', 'N', 'E', 'A', 'S', 'M', 0,
  /* 3116 */ 'G', '_', 'F', 'M', 'I', 'N', 'I', 'M', 'U', 'M', 0,
  /* 3127 */ 'G', '_', 'F', 'M', 'A', 'X', 'I', 'M', 'U', 'M', 0,
  /* 3138 */ 'G', '_', 'F', 'M', 'I', 'N', 'N', 'U', 'M', 0,
  /* 3148 */ 'G', '_', 'F', 'M', 'A', 'X', 'N', 'U', 'M', 0,
  /* 3158 */ 'G', '_', 'I', 'N', 'T', 'R', 'I', 'N', 'S', 'I', 'C', '_', 'R', 'O', 'U', 'N', 'D', 'E', 'V', 'E', 'N', 0,
  /* 3180 */ 'G', '_', 'F', 'C', 'O', 'P', 'Y', 'S', 'I', 'G', 'N', 0,
  /* 3192 */ 'D', 'U', 'M', 'M', 'Y', '_', 'C', 'H', 'A', 'I', 'N', 0,
  /* 3204 */ 'E', 'N', 'D', 'M', 'A', 'I', 'N', 0,
  /* 3212 */ 'G', '_', 'V', 'E', 'C', 'R', 'E', 'D', 'U', 'C', 'E', '_', 'F', 'M', 'I', 'N', 0,
  /* 3229 */ 'G', '_', 'V', 'E', 'C', 'R', 'E', 'D', 'U', 'C', 'E', '_', 'S', 'M', 'I', 'N', 0,
  /* 3246 */ 'G', '_', 'S', 'M', 'I', 'N', 0,
  /* 3253 */ 'G', '_', 'V', 'E', 'C', 'R', 'E', 'D', 'U', 'C', 'E', '_', 'U', 'M', 'I', 'N', 0,
  /* 3270 */ 'G', '_', 'U', 'M', 'I', 'N', 0,
  /* 3277 */ 'G', '_', 'A', 'T', 'O', 'M', 'I', 'C', 'R', 'M', 'W', '_', 'U', 'M', 'I', 'N', 0,
  /* 3294 */ 'G', '_', 'A', 'T', 'O', 'M', 'I', 'C', 'R', 'M', 'W', '_', 'M', 'I', 'N', 0,
  /* 3310 */ 'G', '_', 'F', 'S', 'I', 'N', 0,
  /* 3317 */ 'C', 'F', 'I', '_', 'I', 'N', 'S', 'T', 'R', 'U', 'C', 'T', 'I', 'O', 'N', 0,
  /* 3333 */ 'R', 'E', 'T', 'U', 'R', 'N', 0,
  /* 3340 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'R', 'S', 'U', 'B', '_', 'R', 'T', 'N', 0,
  /* 3360 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'S', 'U', 'B', '_', 'R', 'T', 'N', 0,
  /* 3379 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'A', 'D', 'D', '_', 'R', 'T', 'N', 0,
  /* 3398 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'A', 'N', 'D', '_', 'R', 'T', 'N', 0,
  /* 3417 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'X', 'O', 'R', '_', 'R', 'T', 'N', 0,
  /* 3436 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'O', 'R', '_', 'R', 'T', 'N', 0,
  /* 3454 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'D', 'E', 'C', '_', 'U', 'I', 'N', 'T', '_', 'R', 'T', 'N', 0,
  /* 3478 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'I', 'N', 'C', '_', 'U', 'I', 'N', 'T', '_', 'R', 'T', 'N', 0,
  /* 3502 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'M', 'I', 'N', '_', 'U', 'I', 'N', 'T', '_', 'R', 'T', 'N', 0,
  /* 3526 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'M', 'A', 'X', '_', 'U', 'I', 'N', 'T', '_', 'R', 'T', 'N', 0,
  /* 3550 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'C', 'M', 'P', 'X', 'C', 'H', 'G', '_', 'I', 'N', 'T', '_', 'R', 'T', 'N', 0,
  /* 3577 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'X', 'C', 'H', 'G', '_', 'I', 'N', 'T', '_', 'R', 'T', 'N', 0,
  /* 3601 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'M', 'I', 'N', '_', 'I', 'N', 'T', '_', 'R', 'T', 'N', 0,
  /* 3624 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'M', 'A', 'X', '_', 'I', 'N', 'T', '_', 'R', 'T', 'N', 0,
  /* 3647 */ 'R', 'E', 'T', 'D', 'Y', 'N', 0,
  /* 3654 */ 'G', '_', 'S', 'S', 'U', 'B', 'O', 0,
  /* 3662 */ 'G', '_', 'U', 'S', 'U', 'B', 'O', 0,
  /* 3670 */ 'G', '_', 'S', 'A', 'D', 'D', 'O', 0,
  /* 3678 */ 'G', '_', 'U', 'A', 'D', 'D', 'O', 0,
  /* 3686 */ 'T', 'E', 'X', '_', 'G', 'E', 'T', '_', 'T', 'E', 'X', 'T', 'U', 'R', 'E', '_', 'R', 'E', 'S', 'I', 'N', 'F', 'O', 0,
  /* 3710 */ 'G', '_', 'S', 'M', 'U', 'L', 'O', 0,
  /* 3718 */ 'G', '_', 'U', 'M', 'U', 'L', 'O', 0,
  /* 3726 */ 'S', 'T', 'A', 'C', 'K', 'M', 'A', 'P', 0,
  /* 3735 */ 'G', '_', 'B', 'S', 'W', 'A', 'P', 0,
  /* 3743 */ 'G', '_', 'S', 'I', 'T', 'O', 'F', 'P', 0,
  /* 3752 */ 'G', '_', 'U', 'I', 'T', 'O', 'F', 'P', 0,
  /* 3761 */ 'G', '_', 'F', 'C', 'M', 'P', 0,
  /* 3768 */ 'G', '_', 'I', 'C', 'M', 'P', 0,
  /* 3775 */ 'J', 'U', 'M', 'P', 0,
  /* 3780 */ 'E', 'N', 'D', 'L', 'O', 'O', 'P', 0,
  /* 3788 */ 'W', 'H', 'I', 'L', 'E', 'L', 'O', 'O', 'P', 0,
  /* 3798 */ 'G', '_', 'C', 'T', 'P', 'O', 'P', 0,
  /* 3806 */ 'P', 'A', 'T', 'C', 'H', 'A', 'B', 'L', 'E', '_', 'O', 'P', 0,
  /* 3819 */ 'F', 'A', 'U', 'L', 'T', 'I', 'N', 'G', '_', 'O', 'P', 0,
  /* 3831 */ 'P', 'R', 'E', 'A', 'L', 'L', 'O', 'C', 'A', 'T', 'E', 'D', '_', 'S', 'E', 'T', 'U', 'P', 0,
  /* 3850 */ 'G', '_', 'F', 'E', 'X', 'P', 0,
  /* 3857 */ 'G', '_', 'B', 'R', 0,
  /* 3862 */ 'I', 'N', 'L', 'I', 'N', 'E', 'A', 'S', 'M', '_', 'B', 'R', 0,
  /* 3875 */ 'G', '_', 'B', 'L', 'O', 'C', 'K', '_', 'A', 'D', 'D', 'R', 0,
  /* 3888 */ 'M', 'O', 'V', '_', 'I', 'M', 'M', '_', 'G', 'L', 'O', 'B', 'A', 'L', '_', 'A', 'D', 'D', 'R', 0,
  /* 3908 */ 'G', 'R', 'O', 'U', 'P', '_', 'B', 'A', 'R', 'R', 'I', 'E', 'R', 0,
  /* 3922 */ 'C', 'F', '_', 'A', 'L', 'U', '_', 'E', 'L', 'S', 'E', '_', 'A', 'F', 'T', 'E', 'R', 0,
  /* 3940 */ 'C', 'F', '_', 'A', 'L', 'U', '_', 'P', 'O', 'P', '_', 'A', 'F', 'T', 'E', 'R', 0,
  /* 3957 */ 'P', 'A', 'T', 'C', 'H', 'A', 'B', 'L', 'E', '_', 'F', 'U', 'N', 'C', 'T', 'I', 'O', 'N', '_', 'E', 'N', 'T', 'E', 'R', 0,
  /* 3982 */ 'G', '_', 'R', 'E', 'A', 'D', 'C', 'Y', 'C', 'L', 'E', 'C', 'O', 'U', 'N', 'T', 'E', 'R', 0,
  /* 4001 */ 'G', '_', 'R', 'E', 'A', 'D', '_', 'R', 'E', 'G', 'I', 'S', 'T', 'E', 'R', 0,
  /* 4017 */ 'G', '_', 'W', 'R', 'I', 'T', 'E', '_', 'R', 'E', 'G', 'I', 'S', 'T', 'E', 'R', 0,
  /* 4034 */ 'G', '_', 'A', 'S', 'H', 'R', 0,
  /* 4041 */ 'G', '_', 'F', 'S', 'H', 'R', 0,
  /* 4048 */ 'G', '_', 'L', 'S', 'H', 'R', 0,
  /* 4055 */ 'R', 'A', 'T', '_', 'M', 'S', 'K', 'O', 'R', 0,
  /* 4065 */ 'G', '_', 'F', 'F', 'L', 'O', 'O', 'R', 0,
  /* 4074 */ 'G', '_', 'B', 'U', 'I', 'L', 'D', '_', 'V', 'E', 'C', 'T', 'O', 'R', 0,
  /* 4089 */ 'G', '_', 'S', 'H', 'U', 'F', 'F', 'L', 'E', '_', 'V', 'E', 'C', 'T', 'O', 'R', 0,
  /* 4106 */ 'G', '_', 'V', 'E', 'C', 'R', 'E', 'D', 'U', 'C', 'E', '_', 'X', 'O', 'R', 0,
  /* 4122 */ 'G', '_', 'X', 'O', 'R', 0,
  /* 4128 */ 'L', 'D', 'S', '_', 'X', 'O', 'R', 0,
  /* 4136 */ 'G', '_', 'A', 'T', 'O', 'M', 'I', 'C', 'R', 'M', 'W', '_', 'X', 'O', 'R', 0,
  /* 4152 */ 'G', '_', 'V', 'E', 'C', 'R', 'E', 'D', 'U', 'C', 'E', '_', 'O', 'R', 0,
  /* 4167 */ 'G', '_', 'O', 'R', 0,
  /* 4172 */ 'L', 'D', 'S', '_', 'O', 'R', 0,
  /* 4179 */ 'G', '_', 'A', 'T', 'O', 'M', 'I', 'C', 'R', 'M', 'W', '_', 'O', 'R', 0,
  /* 4194 */ 'T', 'E', 'X', '_', 'L', 'D', 'P', 'T', 'R', 0,
  /* 4204 */ 'G', '_', 'I', 'N', 'T', 'T', 'O', 'P', 'T', 'R', 0,
  /* 4215 */ 'G', '_', 'F', 'A', 'B', 'S', 0,
  /* 4222 */ 'G', '_', 'A', 'B', 'S', 0,
  /* 4228 */ 'G', '_', 'U', 'N', 'M', 'E', 'R', 'G', 'E', '_', 'V', 'A', 'L', 'U', 'E', 'S', 0,
  /* 4245 */ 'G', '_', 'M', 'E', 'R', 'G', 'E', '_', 'V', 'A', 'L', 'U', 'E', 'S', 0,
  /* 4260 */ 'L', 'I', 'T', 'E', 'R', 'A', 'L', 'S', 0,
  /* 4269 */ 'G', '_', 'F', 'C', 'O', 'S', 0,
  /* 4276 */ 'G', '_', 'C', 'O', 'N', 'C', 'A', 'T', '_', 'V', 'E', 'C', 'T', 'O', 'R', 'S', 0,
  /* 4293 */ 'C', 'O', 'P', 'Y', '_', 'T', 'O', '_', 'R', 'E', 'G', 'C', 'L', 'A', 'S', 'S', 0,
  /* 4310 */ 'G', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'C', 'M', 'P', 'X', 'C', 'H', 'G', '_', 'W', 'I', 'T', 'H', '_', 'S', 'U', 'C', 'C', 'E', 'S', 'S', 0,
  /* 4340 */ 'G', '_', 'I', 'N', 'T', 'R', 'I', 'N', 'S', 'I', 'C', '_', 'W', '_', 'S', 'I', 'D', 'E', '_', 'E', 'F', 'F', 'E', 'C', 'T', 'S', 0,
  /* 4367 */ 'G', '_', 'S', 'S', 'U', 'B', 'S', 'A', 'T', 0,
  /* 4377 */ 'G', '_', 'U', 'S', 'U', 'B', 'S', 'A', 'T', 0,
  /* 4387 */ 'G', '_', 'S', 'A', 'D', 'D', 'S', 'A', 'T', 0,
  /* 4397 */ 'G', '_', 'U', 'A', 'D', 'D', 'S', 'A', 'T', 0,
  /* 4407 */ 'G', '_', 'S', 'S', 'H', 'L', 'S', 'A', 'T', 0,
  /* 4417 */ 'G', '_', 'U', 'S', 'H', 'L', 'S', 'A', 'T', 0,
  /* 4427 */ 'G', '_', 'S', 'M', 'U', 'L', 'F', 'I', 'X', 'S', 'A', 'T', 0,
  /* 4440 */ 'G', '_', 'U', 'M', 'U', 'L', 'F', 'I', 'X', 'S', 'A', 'T', 0,
  /* 4453 */ 'G', '_', 'S', 'D', 'I', 'V', 'F', 'I', 'X', 'S', 'A', 'T', 0,
  /* 4466 */ 'G', '_', 'U', 'D', 'I', 'V', 'F', 'I', 'X', 'S', 'A', 'T', 0,
  /* 4479 */ 'F', 'R', 'A', 'C', 'T', 0,
  /* 4485 */ 'G', '_', 'E', 'X', 'T', 'R', 'A', 'C', 'T', 0,
  /* 4495 */ 'G', '_', 'S', 'E', 'L', 'E', 'C', 'T', 0,
  /* 4504 */ 'G', '_', 'B', 'R', 'I', 'N', 'D', 'I', 'R', 'E', 'C', 'T', 0,
  /* 4517 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'R', 'S', 'U', 'B', '_', 'N', 'O', 'R', 'E', 'T', 0,
  /* 4539 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'S', 'U', 'B', '_', 'N', 'O', 'R', 'E', 'T', 0,
  /* 4560 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'A', 'D', 'D', '_', 'N', 'O', 'R', 'E', 'T', 0,
  /* 4581 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'A', 'N', 'D', '_', 'N', 'O', 'R', 'E', 'T', 0,
  /* 4602 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'X', 'O', 'R', '_', 'N', 'O', 'R', 'E', 'T', 0,
  /* 4623 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'O', 'R', '_', 'N', 'O', 'R', 'E', 'T', 0,
  /* 4643 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'D', 'E', 'C', '_', 'U', 'I', 'N', 'T', '_', 'N', 'O', 'R', 'E', 'T', 0,
  /* 4669 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'I', 'N', 'C', '_', 'U', 'I', 'N', 'T', '_', 'N', 'O', 'R', 'E', 'T', 0,
  /* 4695 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'M', 'I', 'N', '_', 'U', 'I', 'N', 'T', '_', 'N', 'O', 'R', 'E', 'T', 0,
  /* 4721 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'M', 'A', 'X', '_', 'U', 'I', 'N', 'T', '_', 'N', 'O', 'R', 'E', 'T', 0,
  /* 4747 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'C', 'M', 'P', 'X', 'C', 'H', 'G', '_', 'I', 'N', 'T', '_', 'N', 'O', 'R', 'E', 'T', 0,
  /* 4776 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'X', 'C', 'H', 'G', '_', 'I', 'N', 'T', '_', 'N', 'O', 'R', 'E', 'T', 0,
  /* 4802 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'M', 'I', 'N', '_', 'I', 'N', 'T', '_', 'N', 'O', 'R', 'E', 'T', 0,
  /* 4827 */ 'R', 'A', 'T', '_', 'A', 'T', 'O', 'M', 'I', 'C', '_', 'M', 'A', 'X', '_', 'I', 'N', 'T', '_', 'N', 'O', 'R', 'E', 'T', 0,
  /* 4852 */ 'L', 'D', 'S', '_', 'S', 'U', 'B', '_', 'R', 'E', 'T', 0,
  /* 4864 */ 'L', 'D', 'S', '_', 'U', 'B', 'Y', 'T', 'E', '_', 'R', 'E', 'A', 'D', '_', 'R', 'E', 'T', 0,
  /* 4883 */ 'L', 'D', 'S', '_', 'B', 'Y', 'T', 'E', '_', 'R', 'E', 'A', 'D', '_', 'R', 'E', 'T', 0,
  /* 4901 */ 'L', 'D', 'S', '_', 'R', 'E', 'A', 'D', '_', 'R', 'E', 'T', 0,
  /* 4914 */ 'L', 'D', 'S', '_', 'U', 'S', 'H', 'O', 'R', 'T', '_', 'R', 'E', 'A', 'D', '_', 'R', 'E', 'T', 0,
  /* 4934 */ 'L', 'D', 'S', '_', 'S', 'H', 'O', 'R', 'T', '_', 'R', 'E', 'A', 'D', '_', 'R', 'E', 'T', 0,
  /* 4953 */ 'L', 'D', 'S', '_', 'A', 'D', 'D', '_', 'R', 'E', 'T', 0,
  /* 4965 */ 'L', 'D', 'S', '_', 'A', 'N', 'D', '_', 'R', 'E', 'T', 0,
  /* 4977 */ 'P', 'A', 'T', 'C', 'H', 'A', 'B', 'L', 'E', '_', 'R', 'E', 'T', 0,
  /* 4991 */ 'L', 'D', 'S', '_', 'W', 'R', 'X', 'C', 'H', 'G', '_', 'R', 'E', 'T', 0,
  /* 5006 */ 'L', 'D', 'S', '_', 'X', 'O', 'R', '_', 'R', 'E', 'T', 0,
  /* 5018 */ 'L', 'D', 'S', '_', 'O', 'R', '_', 'R', 'E', 'T', 0,
  /* 5029 */ 'L', 'D', 'S', '_', 'M', 'I', 'N', '_', 'U', 'I', 'N', 'T', '_', 'R', 'E', 'T', 0,
  /* 5046 */ 'L', 'D', 'S', '_', 'M', 'A', 'X', '_', 'U', 'I', 'N', 'T', '_', 'R', 'E', 'T', 0,
  /* 5063 */ 'L', 'D', 'S', '_', 'M', 'I', 'N', '_', 'I', 'N', 'T', '_', 'R', 'E', 'T', 0,
  /* 5079 */ 'L', 'D', 'S', '_', 'M', 'A', 'X', '_', 'I', 'N', 'T', '_', 'R', 'E', 'T', 0,
  /* 5095 */ 'L', 'D', 'S', '_', 'C', 'M', 'P', 'S', 'T', '_', 'R', 'E', 'T', 0,
  /* 5109 */ 'G', '_', 'M', 'E', 'M', 'S', 'E', 'T', 0,
  /* 5118 */ 'I', 'F', '_', 'P', 'R', 'E', 'D', 'I', 'C', 'A', 'T', 'E', '_', 'S', 'E', 'T', 0,
  /* 5135 */ 'K', 'I', 'L', 'L', 'G', 'T', 0,
  /* 5142 */ 'S', 'G', 'T', 0,
  /* 5146 */ 'P', 'R', 'E', 'D', '_', 'S', 'E', 'T', 'G', 'T', 0,
  /* 5157 */ 'P', 'A', 'T', 'C', 'H', 'A', 'B', 'L', 'E', '_', 'F', 'U', 'N', 'C', 'T', 'I', 'O', 'N', '_', 'E', 'X', 'I', 'T', 0,
  /* 5181 */ 'G', '_', 'B', 'R', 'J', 'T', 0,
  /* 5188 */ 'G', '_', 'E', 'X', 'T', 'R', 'A', 'C', 'T', '_', 'V', 'E', 'C', 'T', 'O', 'R', '_', 'E', 'L', 'T', 0,
  /* 5209 */ 'G', '_', 'I', 'N', 'S', 'E', 'R', 'T', '_', 'V', 'E', 'C', 'T', 'O', 'R', '_', 'E', 'L', 'T', 0,
  /* 5229 */ 'D', 'E', 'F', 'A', 'U', 'L', 'T', 0,
  /* 5237 */ 'G', '_', 'F', 'C', 'O', 'N', 'S', 'T', 'A', 'N', 'T', 0,
  /* 5249 */ 'G', '_', 'C', 'O', 'N', 'S', 'T', 'A', 'N', 'T', 0,
  /* 5260 */ 'S', 'T', 'A', 'T', 'E', 'P', 'O', 'I', 'N', 'T', 0,
  /* 5271 */ 'P', 'A', 'T', 'C', 'H', 'P', 'O', 'I', 'N', 'T', 0,
  /* 5282 */ 'G', '_', 'P', 'T', 'R', 'T', 'O', 'I', 'N', 'T', 0,
  /* 5293 */ 'G', '_', 'F', 'R', 'I', 'N', 'T', 0,
  /* 5301 */ 'G', '_', 'I', 'N', 'T', 'R', 'I', 'N', 'S', 'I', 'C', '_', 'L', 'R', 'I', 'N', 'T', 0,
  /* 5319 */ 'S', 'U', 'B', 'B', '_', 'U', 'I', 'N', 'T', 0,
  /* 5329 */ 'A', 'D', 'D', 'C', '_', 'U', 'I', 'N', 'T', 0,
  /* 5339 */ 'S', 'E', 'T', 'G', 'E', '_', 'U', 'I', 'N', 'T', 0,
  /* 5350 */ 'F', 'F', 'B', 'H', '_', 'U', 'I', 'N', 'T', 0,
  /* 5360 */ 'L', 'D', 'S', '_', 'M', 'I', 'N', '_', 'U', 'I', 'N', 'T', 0,
  /* 5373 */ 'S', 'E', 'T', 'G', 'T', '_', 'U', 'I', 'N', 'T', 0,
  /* 5384 */ 'L', 'D', 'S', '_', 'M', 'A', 'X', '_', 'U', 'I', 'N', 'T', 0,
  /* 5397 */ 'G', '_', 'F', 'N', 'E', 'A', 'R', 'B', 'Y', 'I', 'N', 'T', 0,
  /* 5410 */ 'S', 'U', 'B', '_', 'I', 'N', 'T', 0,
  /* 5418 */ 'A', 'D', 'D', '_', 'I', 'N', 'T', 0,
  /* 5426 */ 'A', 'N', 'D', '_', 'I', 'N', 'T', 0,
  /* 5434 */ 'C', 'N', 'D', 'E', '_', 'I', 'N', 'T', 0,
  /* 5443 */ 'C', 'N', 'D', 'G', 'E', '_', 'I', 'N', 'T', 0,
  /* 5453 */ 'P', 'R', 'E', 'D', '_', 'S', 'E', 'T', 'G', 'E', '_', 'I', 'N', 'T', 0,
  /* 5468 */ 'P', 'R', 'E', 'D', '_', 'S', 'E', 'T', 'N', 'E', '_', 'I', 'N', 'T', 0,
  /* 5483 */ 'P', 'R', 'E', 'D', '_', 'S', 'E', 'T', 'E', '_', 'I', 'N', 'T', 0,
  /* 5497 */ 'F', 'F', 'B', 'L', '_', 'I', 'N', 'T', 0,
  /* 5506 */ 'L', 'D', 'S', '_', 'M', 'I', 'N', '_', 'I', 'N', 'T', 0,
  /* 5518 */ 'X', 'O', 'R', '_', 'I', 'N', 'T', 0,
  /* 5526 */ 'C', 'N', 'D', 'G', 'T', '_', 'I', 'N', 'T', 0,
  /* 5536 */ 'P', 'R', 'E', 'D', '_', 'S', 'E', 'T', 'G', 'T', '_', 'I', 'N', 'T', 0,
  /* 5551 */ 'B', 'C', 'N', 'T', '_', 'I', 'N', 'T', 0,
  /* 5560 */ 'N', 'O', 'T', '_', 'I', 'N', 'T', 0,
  /* 5568 */ 'L', 'D', 'S', '_', 'M', 'A', 'X', '_', 'I', 'N', 'T', 0,
  /* 5580 */ 'G', '_', 'V', 'A', 'S', 'T', 'A', 'R', 'T', 0,
  /* 5590 */ 'L', 'I', 'F', 'E', 'T', 'I', 'M', 'E', '_', 'S', 'T', 'A', 'R', 'T', 0,
  /* 5605 */ 'G', '_', 'I', 'N', 'S', 'E', 'R', 'T', 0,
  /* 5614 */ 'G', '_', 'F', 'S', 'Q', 'R', 'T', 0,
  /* 5622 */ 'G', '_', 'S', 'T', 'R', 'I', 'C', 'T', '_', 'F', 'S', 'Q', 'R', 'T', 0,
  /* 5637 */ 'G', '_', 'B', 'I', 'T', 'C', 'A', 'S', 'T', 0,
  /* 5647 */ 'G', '_', 'A', 'D', 'D', 'R', 'S', 'P', 'A', 'C', 'E', '_', 'C', 'A', 'S', 'T', 0,
  /* 5664 */ 'L', 'D', 'S', '_', 'C', 'M', 'P', 'S', 'T', 0,
  /* 5674 */ 'G', '_', 'F', 'P', 'E', 'X', 'T', 0,
  /* 5682 */ 'G', '_', 'S', 'E', 'X', 'T', 0,
  /* 5689 */ 'G', '_', 'A', 'N', 'Y', 'E', 'X', 'T', 0,
  /* 5698 */ 'G', '_', 'Z', 'E', 'X', 'T', 0,
  /* 5705 */ 'C', 'F', '_', 'A', 'L', 'U', 0,
  /* 5712 */ 'G', '_', 'F', 'D', 'I', 'V', 0,
  /* 5719 */ 'G', '_', 'S', 'T', 'R', 'I', 'C', 'T', '_', 'F', 'D', 'I', 'V', 0,
  /* 5733 */ 'G', '_', 'S', 'D', 'I', 'V', 0,
  /* 5740 */ 'G', '_', 'U', 'D', 'I', 'V', 0,
  /* 5747 */ 'M', 'O', 'V', 0,
  /* 5751 */ 'T', 'E', 'X', '_', 'G', 'E', 'T', '_', 'G', 'R', 'A', 'D', 'I', 'E', 'N', 'T', 'S', '_', 'V', 0,
  /* 5771 */ 'T', 'E', 'X', '_', 'S', 'E', 'T', '_', 'G', 'R', 'A', 'D', 'I', 'E', 'N', 'T', 'S', '_', 'V', 0,
  /* 5791 */ 'T', 'X', 'D', '_', 'S', 'H', 'A', 'D', 'O', 'W', 0,
  /* 5802 */ 'G', '_', 'F', 'P', 'O', 'W', 0,
  /* 5809 */ 'I', 'N', 'T', 'E', 'R', 'P', '_', 'Z', 'W', 0,
  /* 5819 */ 'I', 'N', 'T', 'E', 'R', 'P', '_', 'P', 'A', 'I', 'R', '_', 'Z', 'W', 0,
  /* 5834 */ 'G', '_', 'V', 'E', 'C', 'R', 'E', 'D', 'U', 'C', 'E', '_', 'F', 'M', 'A', 'X', 0,
  /* 5851 */ 'G', '_', 'V', 'E', 'C', 'R', 'E', 'D', 'U', 'C', 'E', '_', 'S', 'M', 'A', 'X', 0,
  /* 5868 */ 'G', '_', 'S', 'M', 'A', 'X', 0,
  /* 5875 */ 'G', '_', 'V', 'E', 'C', 'R', 'E', 'D', 'U', 'C', 'E', '_', 'U', 'M', 'A', 'X', 0,
  /* 5892 */ 'G', '_', 'U', 'M', 'A', 'X', 0,
  /* 5899 */ 'G', '_', 'A', 'T', 'O', 'M', 'I', 'C', 'R', 'M', 'W', '_', 'U', 'M', 'A', 'X', 0,
  /* 5916 */ 'G', '_', 'A', 'T', 'O', 'M', 'I', 'C', 'R', 'M', 'W', '_', 'M', 'A', 'X', 0,
  /* 5932 */ 'G', '_', 'F', 'R', 'A', 'M', 'E', '_', 'I', 'N', 'D', 'E', 'X', 0,
  /* 5946 */ 'G', '_', 'S', 'M', 'U', 'L', 'F', 'I', 'X', 0,
  /* 5956 */ 'G', '_', 'U', 'M', 'U', 'L', 'F', 'I', 'X', 0,
  /* 5966 */ 'G', '_', 'S', 'D', 'I', 'V', 'F', 'I', 'X', 0,
  /* 5976 */ 'G', '_', 'U', 'D', 'I', 'V', 'F', 'I', 'X', 0,
  /* 5986 */ 'P', 'R', 'E', 'D', '_', 'X', 0,
  /* 5993 */ 'G', '_', 'M', 'E', 'M', 'C', 'P', 'Y', 0,
  /* 6002 */ 'C', 'O', 'N', 'S', 'T', '_', 'C', 'O', 'P', 'Y', 0,
  /* 6013 */ 'I', 'N', 'T', 'E', 'R', 'P', '_', 'X', 'Y', 0,
  /* 6023 */ 'I', 'N', 'T', 'E', 'R', 'P', '_', 'P', 'A', 'I', 'R', '_', 'X', 'Y', 0,
  /* 6038 */ 'G', '_', 'C', 'T', 'L', 'Z', 0,
  /* 6045 */ 'G', '_', 'C', 'T', 'T', 'Z', 0,
  /* 6052 */ 'R', '6', '0', '0', '_', 'R', 'e', 'g', 'i', 's', 't', 'e', 'r', 'L', 'o', 'a', 'd', 0,
  /* 6070 */ 'R', '6', '0', '0', '_', 'R', 'e', 'g', 'i', 's', 't', 'e', 'r', 'S', 't', 'o', 'r', 'e', 0,
  /* 6089 */ 'R', '6', '0', '0', '_', 'E', 'x', 'p', 'o', 'r', 't', 'B', 'u', 'f', 0,
  /* 6104 */ 'E', 'G', '_', 'E', 'x', 'p', 'o', 'r', 't', 'B', 'u', 'f', 0,
  /* 6117 */ 'V', 'T', 'X', '_', 'R', 'E', 'A', 'D', '_', '3', '2', '_', 'e', 'g', 0,
  /* 6132 */ 'R', 'A', 'T', '_', 'W', 'R', 'I', 'T', 'E', '_', 'C', 'A', 'C', 'H', 'E', 'L', 'E', 'S', 'S', '_', '3', '2', '_', 'e', 'g', 0,
  /* 6158 */ 'M', 'U', 'L', 'A', 'D', 'D', '_', 'U', 'I', 'N', 'T', '2', '4', '_', 'e', 'g', 0,
  /* 6175 */ 'M', 'U', 'L', 'H', 'I', '_', 'U', 'I', 'N', 'T', '2', '4', '_', 'e', 'g', 0,
  /* 6191 */ 'M', 'U', 'L', '_', 'U', 'I', 'N', 'T', '2', '4', '_', 'e', 'g', 0,
  /* 6205 */ 'V', 'T', 'X', '_', 'R', 'E', 'A', 'D', '_', '6', '4', '_', 'e', 'g', 0,
  /* 6220 */ 'R', 'A', 'T', '_', 'W', 'R', 'I', 'T', 'E', '_', 'C', 'A', 'C', 'H', 'E', 'L', 'E', 'S', 'S', '_', '6', '4', '_', 'e', 'g', 0,
  /* 6246 */ 'D', 'O', 'T', '4', '_', 'e', 'g', 0,
  /* 6254 */ 'V', 'T', 'X', '_', 'R', 'E', 'A', 'D', '_', '1', '6', '_', 'e', 'g', 0,
  /* 6269 */ 'V', 'T', 'X', '_', 'R', 'E', 'A', 'D', '_', '1', '2', '8', '_', 'e', 'g', 0,
  /* 6285 */ 'R', 'A', 'T', '_', 'W', 'R', 'I', 'T', 'E', '_', 'C', 'A', 'C', 'H', 'E', 'L', 'E', 'S', 'S', '_', '1', '2', '8', '_', 'e', 'g', 0,
  /* 6312 */ 'V', 'T', 'X', '_', 'R', 'E', 'A', 'D', '_', '8', '_', 'e', 'g', 0,
  /* 6326 */ 'F', 'M', 'A', '_', 'e', 'g', 0,
  /* 6333 */ 'M', 'U', 'L', 'A', 'D', 'D', '_', 'e', 'g', 0,
  /* 6343 */ 'L', 'O', 'G', '_', 'C', 'L', 'A', 'M', 'P', 'E', 'D', '_', 'e', 'g', 0,
  /* 6358 */ 'R', 'E', 'C', 'I', 'P', '_', 'C', 'L', 'A', 'M', 'P', 'E', 'D', '_', 'e', 'g', 0,
  /* 6375 */ 'R', 'E', 'C', 'I', 'P', 'S', 'Q', 'R', 'T', '_', 'C', 'L', 'A', 'M', 'P', 'E', 'D', '_', 'e', 'g', 0,
  /* 6396 */ 'R', 'A', 'T', '_', 'S', 'T', 'O', 'R', 'E', '_', 'T', 'Y', 'P', 'E', 'D', '_', 'e', 'g', 0,
  /* 6415 */ 'C', 'N', 'D', 'E', '_', 'e', 'g', 0,
  /* 6423 */ 'M', 'U', 'L', 'A', 'D', 'D', '_', 'I', 'E', 'E', 'E', '_', 'e', 'g', 0,
  /* 6438 */ 'L', 'O', 'G', '_', 'I', 'E', 'E', 'E', '_', 'e', 'g', 0,
  /* 6450 */ 'R', 'E', 'C', 'I', 'P', '_', 'I', 'E', 'E', 'E', '_', 'e', 'g', 0,
  /* 6464 */ 'E', 'X', 'P', '_', 'I', 'E', 'E', 'E', '_', 'e', 'g', 0,
  /* 6476 */ 'R', 'E', 'C', 'I', 'P', 'S', 'Q', 'R', 'T', '_', 'I', 'E', 'E', 'E', '_', 'e', 'g', 0,
  /* 6494 */ 'C', 'N', 'D', 'G', 'E', '_', 'e', 'g', 0,
  /* 6503 */ 'L', 'S', 'H', 'L', '_', 'e', 'g', 0,
  /* 6511 */ 'S', 'I', 'N', '_', 'e', 'g', 0,
  /* 6518 */ 'A', 'S', 'H', 'R', '_', 'e', 'g', 0,
  /* 6526 */ 'L', 'S', 'H', 'R', '_', 'e', 'g', 0,
  /* 6534 */ 'C', 'O', 'S', '_', 'e', 'g', 0,
  /* 6541 */ 'C', 'N', 'D', 'G', 'T', '_', 'e', 'g', 0,
  /* 6550 */ 'M', 'U', 'L', '_', 'L', 'I', 'T', '_', 'e', 'g', 0,
  /* 6561 */ 'U', 'I', 'N', 'T', '_', 'T', 'O', '_', 'F', 'L', 'T', '_', 'e', 'g', 0,
  /* 6576 */ 'B', 'F', 'E', '_', 'U', 'I', 'N', 'T', '_', 'e', 'g', 0,
  /* 6588 */ 'M', 'U', 'L', 'H', 'I', '_', 'U', 'I', 'N', 'T', '_', 'e', 'g', 0,
  /* 6602 */ 'M', 'U', 'L', 'L', 'O', '_', 'U', 'I', 'N', 'T', '_', 'e', 'g', 0,
  /* 6616 */ 'F', 'L', 'T', '_', 'T', 'O', '_', 'U', 'I', 'N', 'T', '_', 'e', 'g', 0,
  /* 6631 */ 'R', 'E', 'C', 'I', 'P', '_', 'U', 'I', 'N', 'T', '_', 'e', 'g', 0,
  /* 6645 */ 'M', 'O', 'V', 'A', '_', 'I', 'N', 'T', '_', 'e', 'g', 0,
  /* 6657 */ 'B', 'F', 'E', '_', 'I', 'N', 'T', '_', 'e', 'g', 0,
  /* 6668 */ 'B', 'F', 'I', '_', 'I', 'N', 'T', '_', 'e', 'g', 0,
  /* 6679 */ 'M', 'U', 'L', 'H', 'I', '_', 'I', 'N', 'T', '_', 'e', 'g', 0,
  /* 6692 */ 'B', 'F', 'M', '_', 'I', 'N', 'T', '_', 'e', 'g', 0,
  /* 6703 */ 'B', 'I', 'T', '_', 'A', 'L', 'I', 'G', 'N', '_', 'I', 'N', 'T', '_', 'e', 'g', 0,
  /* 6720 */ 'M', 'U', 'L', 'L', 'O', '_', 'I', 'N', 'T', '_', 'e', 'g', 0,
  /* 6733 */ 'F', 'L', 'T', '_', 'T', 'O', '_', 'I', 'N', 'T', '_', 'e', 'g', 0,
  /* 6747 */ 'C', 'U', 'B', 'E', '_', 'r', '6', '0', '0', '_', 'r', 'e', 'a', 'l', 0,
  /* 6762 */ 'C', 'U', 'B', 'E', '_', 'e', 'g', '_', 'r', 'e', 'a', 'l', 0,
  /* 6775 */ 'V', 'T', 'X', '_', 'R', 'E', 'A', 'D', '_', '3', '2', '_', 'c', 'm', 0,
  /* 6790 */ 'M', 'U', 'L', 'A', 'D', 'D', '_', 'I', 'N', 'T', '2', '4', '_', 'c', 'm', 0,
  /* 6806 */ 'M', 'U', 'L', '_', 'I', 'N', 'T', '2', '4', '_', 'c', 'm', 0,
  /* 6819 */ 'V', 'T', 'X', '_', 'R', 'E', 'A', 'D', '_', '6', '4', '_', 'c', 'm', 0,
  /* 6834 */ 'V', 'T', 'X', '_', 'R', 'E', 'A', 'D', '_', '1', '6', '_', 'c', 'm', 0,
  /* 6849 */ 'V', 'T', 'X', '_', 'R', 'E', 'A', 'D', '_', '1', '2', '8', '_', 'c', 'm', 0,
  /* 6865 */ 'V', 'T', 'X', '_', 'R', 'E', 'A', 'D', '_', '8', '_', 'c', 'm', 0,
  /* 6879 */ 'R', 'E', 'C', 'I', 'P', '_', 'C', 'L', 'A', 'M', 'P', 'E', 'D', '_', 'c', 'm', 0,
  /* 6896 */ 'R', 'E', 'C', 'I', 'P', 'S', 'Q', 'R', 'T', '_', 'C', 'L', 'A', 'M', 'P', 'E', 'D', '_', 'c', 'm', 0,
  /* 6917 */ 'R', 'A', 'T', '_', 'S', 'T', 'O', 'R', 'E', '_', 'T', 'Y', 'P', 'E', 'D', '_', 'c', 'm', 0,
  /* 6936 */ 'L', 'O', 'G', '_', 'I', 'E', 'E', 'E', '_', 'c', 'm', 0,
  /* 6948 */ 'R', 'E', 'C', 'I', 'P', '_', 'I', 'E', 'E', 'E', '_', 'c', 'm', 0,
  /* 6962 */ 'E', 'X', 'P', '_', 'I', 'E', 'E', 'E', '_', 'c', 'm', 0,
  /* 6974 */ 'R', 'E', 'C', 'I', 'P', 'S', 'Q', 'R', 'T', '_', 'I', 'E', 'E', 'E', '_', 'c', 'm', 0,
  /* 6992 */ 'S', 'I', 'N', '_', 'c', 'm', 0,
  /* 6999 */ 'C', 'O', 'S', '_', 'c', 'm', 0,
  /* 7006 */ 'M', 'U', 'L', 'H', 'I', '_', 'U', 'I', 'N', 'T', '_', 'c', 'm', 0,
  /* 7020 */ 'M', 'U', 'L', 'L', 'O', '_', 'U', 'I', 'N', 'T', '_', 'c', 'm', 0,
  /* 7034 */ 'M', 'U', 'L', 'H', 'I', '_', 'I', 'N', 'T', '_', 'c', 'm', 0,
  /* 7047 */ 'M', 'U', 'L', 'L', 'O', '_', 'I', 'N', 'T', '_', 'c', 'm', 0,
  /* 7060 */ 'C', 'U', 'B', 'E', '_', 'r', '6', '0', '0', '_', 'p', 's', 'e', 'u', 'd', 'o', 0,
  /* 7077 */ 'C', 'U', 'B', 'E', '_', 'e', 'g', '_', 'p', 's', 'e', 'u', 'd', 'o', 0,
  /* 7092 */ 'R', '6', '0', '0', '_', 'E', 'x', 'p', 'o', 'r', 't', 'S', 'w', 'z', 0,
  /* 7107 */ 'E', 'G', '_', 'E', 'x', 'p', 'o', 'r', 't', 'S', 'w', 'z', 0,

};

extern const unsigned R600InstrNameIndices[] = {
    2728U, 3106U, 3862U, 3317U, 2800U, 2781U, 2809U, 2947U, 
    2355U, 2370U, 2283U, 2397U, 4293U, 2169U, 2296U, 2790U, 
    1874U, 6008U, 1970U, 5590U, 1779U, 3726U, 2935U, 5271U, 
    1829U, 3831U, 2608U, 5260U, 2009U, 3819U, 3806U, 3957U, 
    4977U, 5157U, 2867U, 2914U, 2887U, 2826U, 1669U, 1331U, 
    3027U, 5733U, 5740U, 3092U, 3099U, 1749U, 4167U, 4122U, 
    2281U, 2726U, 5932U, 2179U, 4485U, 4228U, 5605U, 4245U, 
    4074U, 1417U, 4276U, 5282U, 4204U, 5637U, 2220U, 1391U, 
    1811U, 5301U, 3158U, 3982U, 1566U, 1494U, 1524U, 1551U, 
    1475U, 1505U, 2057U, 2041U, 4310U, 2548U, 2576U, 1693U, 
    1345U, 1763U, 1716U, 4179U, 4136U, 5916U, 3294U, 5899U, 
    3277U, 1636U, 1314U, 1866U, 1792U, 4504U, 1361U, 4340U, 
    5689U, 1409U, 5249U, 5237U, 5580U, 2600U, 5682U, 2384U, 
    5698U, 2853U, 4048U, 4034U, 2846U, 4041U, 3768U, 3761U, 
    4495U, 3678U, 1895U, 3662U, 1858U, 3670U, 1887U, 3654U, 
    1850U, 3718U, 3710U, 2678U, 2670U, 4397U, 4387U, 4377U, 
    4367U, 4417U, 4407U, 5946U, 5956U, 4427U, 4440U, 5966U, 
    5976U, 4453U, 4466U, 1594U, 1293U, 2969U, 1244U, 1468U, 
    5712U, 3071U, 5802U, 2750U, 3850U, 1069U, 2593U, 1061U, 
    587U, 2348U, 5674U, 1381U, 2732U, 2741U, 3743U, 3752U, 
    4215U, 3180U, 2229U, 3138U, 3148U, 1912U, 1927U, 3116U, 
    3127U, 1675U, 2771U, 3246U, 5868U, 3270U, 5892U, 4222U, 
    3857U, 5181U, 5209U, 5188U, 4089U, 6045U, 2263U, 6038U, 
    2245U, 3798U, 3735U, 2070U, 2859U, 4269U, 3310U, 5614U, 
    4065U, 5293U, 5397U, 5647U, 3875U, 1957U, 1438U, 1622U, 
    1300U, 2997U, 5719U, 3078U, 1250U, 5622U, 4001U, 4017U, 
    5993U, 2210U, 5109U, 1601U, 2976U, 1577U, 2952U, 5834U, 
    3212U, 1653U, 3011U, 1733U, 4152U, 4106U, 5851U, 3229U, 
    5875U, 3253U, 2653U, 762U, 928U, 2765U, 751U, 917U, 
    875U, 1041U, 816U, 982U, 6002U, 2201U, 729U, 895U, 
    835U, 1001U, 778U, 944U, 7077U, 7060U, 5229U, 1204U, 
    3192U, 2065U, 1788U, 1373U, 2310U, 3780U, 3204U, 2660U, 
    160U, 82U, 1376U, 743U, 909U, 858U, 1024U, 800U, 
    966U, 5118U, 3775U, 1801U, 2132U, 690U, 3888U, 702U, 
    5986U, 1077U, 1165U, 1097U, 1185U, 6052U, 6070U, 3647U, 
    3333U, 1846U, 5791U, 3788U, 1590U, 5329U, 5418U, 2096U, 
    5426U, 6518U, 388U, 5551U, 6657U, 6576U, 6668U, 6692U, 
    6703U, 2862U, 5705U, 2758U, 2194U, 3922U, 3940U, 2022U, 
    2534U, 170U, 2450U, 65U, 2439U, 34U, 3061U, 2429U, 
    22U, 2490U, 108U, 2465U, 47U, 2411U, 0U, 2420U, 
    11U, 5434U, 6415U, 267U, 5443U, 6494U, 358U, 5526U, 
    6541U, 417U, 6999U, 6534U, 408U, 578U, 6762U, 6747U, 
    6246U, 186U, 6104U, 7107U, 2501U, 121U, 6962U, 6464U, 
    324U, 2083U, 5350U, 5497U, 4068U, 714U, 1210U, 6733U, 
    553U, 6616U, 490U, 6326U, 4479U, 3908U, 657U, 6023U, 
    5819U, 1535U, 6013U, 5809U, 6562U, 442U, 5135U, 1685U, 
    4953U, 1755U, 4965U, 4883U, 2117U, 5664U, 5095U, 5568U, 
    5079U, 5384U, 5046U, 5506U, 5063U, 5360U, 5029U, 4172U, 
    5018U, 4901U, 4934U, 2153U, 1337U, 4852U, 4864U, 4914U, 
    2143U, 2565U, 4991U, 4128U, 5006U, 4260U, 6343U, 208U, 
    6936U, 6438U, 294U, 2476U, 92U, 6503U, 369U, 6526U, 
    398U, 5847U, 648U, 5572U, 5388U, 3225U, 628U, 5510U, 
    5364U, 5747U, 6645U, 2965U, 6423U, 277U, 6790U, 6158U, 
    6333U, 196U, 7034U, 1132U, 6679U, 523U, 6175U, 7006U, 
    1116U, 6588U, 458U, 7047U, 6720U, 538U, 7020U, 6602U, 
    474U, 1903U, 6806U, 6550U, 428U, 6191U, 5560U, 5519U, 
    1573U, 2527U, 151U, 2107U, 5483U, 1946U, 5453U, 5146U, 
    5536U, 1998U, 5468U, 6089U, 7092U, 4560U, 3379U, 4581U, 
    3398U, 4747U, 3550U, 4643U, 3454U, 4669U, 3478U, 4827U, 
    3624U, 4721U, 3526U, 4802U, 3601U, 4695U, 3502U, 4623U, 
    3436U, 4517U, 3340U, 4539U, 3360U, 4776U, 3577U, 4602U, 
    3417U, 4055U, 1225U, 672U, 1147U, 6917U, 6396U, 6285U, 
    6132U, 6220U, 6896U, 6375U, 244U, 6974U, 6476U, 338U, 
    6879U, 6358U, 225U, 6948U, 6450U, 308U, 6631U, 507U, 
    1988U, 2112U, 618U, 5488U, 596U, 5458U, 5339U, 637U, 
    5541U, 5373U, 607U, 5473U, 1942U, 5142U, 6992U, 6511U, 
    379U, 569U, 1994U, 5319U, 5410U, 2686U, 5751U, 3686U, 
    1709U, 4194U, 1977U, 1455U, 2625U, 3033U, 1263U, 2640U, 
    3048U, 1279U, 2706U, 5771U, 2316U, 2333U, 1385U, 6561U, 
    441U, 6849U, 6269U, 6834U, 6254U, 6775U, 6117U, 6819U, 
    6205U, 6865U, 6312U, 2513U, 135U, 5518U, 
};

static inline void InitR600MCInstrInfo(MCInstrInfo *II) {
  II->InitMCInstrInfo(R600Insts, R600InstrNameIndices, R600InstrNameData, nullptr, nullptr, 566);
}

} // end namespace llvm
#endif // GET_INSTRINFO_MC_DESC

#ifdef GET_INSTRINFO_HEADER
#undef GET_INSTRINFO_HEADER
namespace llvm {
struct R600GenInstrInfo : public TargetInstrInfo {
  explicit R600GenInstrInfo(int CFSetupOpcode = -1, int CFDestroyOpcode = -1, int CatchRetOpcode = -1, int ReturnOpcode = -1);
  ~R600GenInstrInfo() override = default;

};
} // end namespace llvm
#endif // GET_INSTRINFO_HEADER

#ifdef GET_INSTRINFO_HELPER_DECLS
#undef GET_INSTRINFO_HELPER_DECLS


#endif // GET_INSTRINFO_HELPER_DECLS

#ifdef GET_INSTRINFO_HELPERS
#undef GET_INSTRINFO_HELPERS

#endif // GET_INSTRINFO_HELPERS

#ifdef GET_INSTRINFO_CTOR_DTOR
#undef GET_INSTRINFO_CTOR_DTOR
namespace llvm {
extern const MCInstrDesc R600Insts[];
extern const unsigned R600InstrNameIndices[];
extern const char R600InstrNameData[];
R600GenInstrInfo::R600GenInstrInfo(int CFSetupOpcode, int CFDestroyOpcode, int CatchRetOpcode, int ReturnOpcode)
  : TargetInstrInfo(CFSetupOpcode, CFDestroyOpcode, CatchRetOpcode, ReturnOpcode) {
  InitMCInstrInfo(R600Insts, R600InstrNameIndices, R600InstrNameData, nullptr, nullptr, 566);
}
} // end namespace llvm
#endif // GET_INSTRINFO_CTOR_DTOR

#ifdef GET_INSTRINFO_OPERAND_ENUM
#undef GET_INSTRINFO_OPERAND_ENUM
namespace llvm {
namespace R600 {
namespace OpName {
enum {
  ADDR = 98,
  COUNT = 105,
  Enabled = 106,
  KCACHE_ADDR0 = 103,
  KCACHE_ADDR1 = 104,
  KCACHE_BANK0 = 99,
  KCACHE_BANK1 = 100,
  KCACHE_MODE0 = 101,
  KCACHE_MODE1 = 102,
  addr = 72,
  bank_swizzle = 93,
  chan = 73,
  clamp = 80,
  clamp_W = 58,
  clamp_X = 7,
  clamp_Y = 24,
  clamp_Z = 41,
  dst = 0,
  dst_rel = 79,
  dst_rel_W = 57,
  dst_rel_X = 6,
  dst_rel_Y = 23,
  dst_rel_Z = 40,
  last = 90,
  literal = 92,
  literal0 = 70,
  literal1 = 71,
  omod = 78,
  omod_W = 56,
  omod_X = 5,
  omod_Y = 22,
  omod_Z = 39,
  pred_sel = 91,
  pred_sel_W = 69,
  pred_sel_X = 18,
  pred_sel_Y = 35,
  pred_sel_Z = 52,
  src0 = 1,
  src0_W = 59,
  src0_X = 8,
  src0_Y = 25,
  src0_Z = 42,
  src0_abs = 83,
  src0_abs_W = 62,
  src0_abs_X = 11,
  src0_abs_Y = 28,
  src0_abs_Z = 45,
  src0_neg = 81,
  src0_neg_W = 60,
  src0_neg_X = 9,
  src0_neg_Y = 26,
  src0_neg_Z = 43,
  src0_rel = 82,
  src0_rel_W = 61,
  src0_rel_X = 10,
  src0_rel_Y = 27,
  src0_rel_Z = 44,
  src0_sel = 84,
  src0_sel_W = 63,
  src0_sel_X = 12,
  src0_sel_Y = 29,
  src0_sel_Z = 46,
  src1 = 85,
  src1_W = 64,
  src1_X = 13,
  src1_Y = 30,
  src1_Z = 47,
  src1_abs = 88,
  src1_abs_W = 67,
  src1_abs_X = 16,
  src1_abs_Y = 33,
  src1_abs_Z = 50,
  src1_neg = 86,
  src1_neg_W = 65,
  src1_neg_X = 14,
  src1_neg_Y = 31,
  src1_neg_Z = 48,
  src1_rel = 87,
  src1_rel_W = 66,
  src1_rel_X = 15,
  src1_rel_Y = 32,
  src1_rel_Z = 49,
  src1_sel = 89,
  src1_sel_W = 68,
  src1_sel_X = 17,
  src1_sel_Y = 34,
  src1_sel_Z = 51,
  src2 = 94,
  src2_neg = 95,
  src2_rel = 96,
  src2_sel = 97,
  update_exec_mask = 75,
  update_exec_mask_W = 53,
  update_exec_mask_X = 2,
  update_exec_mask_Y = 19,
  update_exec_mask_Z = 36,
  update_pred = 76,
  update_pred_W = 54,
  update_pred_X = 3,
  update_pred_Y = 20,
  update_pred_Z = 37,
  val = 74,
  write = 77,
  write_W = 55,
  write_X = 4,
  write_Y = 21,
  write_Z = 38,
OPERAND_LAST
};
} // end namespace OpName
} // end namespace R600
} // end namespace llvm
#endif //GET_INSTRINFO_OPERAND_ENUM

#ifdef GET_INSTRINFO_NAMED_OPS
#undef GET_INSTRINFO_NAMED_OPS
namespace llvm {
namespace R600 {
LLVM_READONLY
int16_t getNamedOperandIdx(uint16_t Opcode, uint16_t NamedIdx) {
  static const int16_t OperandMap [][107] = {
{0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, },
{0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, 3, 4, -1, 5, -1, 6, 7, 8, -1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, },
{0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, 3, 4, -1, 5, -1, 6, 10, 11, -1, 12, 7, -1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, },
{0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, 3, -1, -1, -1, -1, -1, 4, 5, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, },
{0, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 4, 5, -1, 6, 7, 8, 9, -1, 10, 15, 16, 17, 18, 11, 12, 13, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, },
{0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 3, 4, 6, 7, 8, 9, -1, -1, -1, -1, -1, 10, 11, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, },
{0, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, },
{0, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, },
{0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, },
{-1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 2, 3, -1, 4, -1, 5, 6, 7, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, },
{-1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 2, 3, -1, 4, -1, 5, 9, 10, -1, 11, 6, -1, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, },
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 3, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, },
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, },
};
  switch(Opcode) {
  case R600::CUBE_eg_pseudo:
  case R600::CUBE_r600_pseudo:
    return OperandMap[0][NamedIdx];
  case R600::LDS_ADD_RET:
  case R600::LDS_AND_RET:
  case R600::LDS_MAX_INT_RET:
  case R600::LDS_MAX_UINT_RET:
  case R600::LDS_MIN_INT_RET:
  case R600::LDS_MIN_UINT_RET:
  case R600::LDS_OR_RET:
  case R600::LDS_SUB_RET:
  case R600::LDS_WRXCHG_RET:
  case R600::LDS_XOR_RET:
    return OperandMap[1][NamedIdx];
  case R600::LDS_CMPST_RET:
    return OperandMap[2][NamedIdx];
  case R600::LDS_BYTE_READ_RET:
  case R600::LDS_READ_RET:
  case R600::LDS_SHORT_READ_RET:
  case R600::LDS_UBYTE_READ_RET:
  case R600::LDS_USHORT_READ_RET:
    return OperandMap[3][NamedIdx];
  case R600::BFE_INT_eg:
  case R600::BFE_UINT_eg:
  case R600::BFI_INT_eg:
  case R600::BIT_ALIGN_INT_eg:
  case R600::CNDE_INT:
  case R600::CNDE_eg:
  case R600::CNDE_r600:
  case R600::CNDGE_INT:
  case R600::CNDGE_eg:
  case R600::CNDGE_r600:
  case R600::CNDGT_INT:
  case R600::CNDGT_eg:
  case R600::CNDGT_r600:
  case R600::FMA_eg:
  case R600::MULADD_IEEE_eg:
  case R600::MULADD_IEEE_r600:
  case R600::MULADD_INT24_cm:
  case R600::MULADD_UINT24_eg:
  case R600::MULADD_eg:
  case R600::MULADD_r600:
  case R600::MUL_LIT_eg:
  case R600::MUL_LIT_r600:
    return OperandMap[4][NamedIdx];
  case R600::BCNT_INT:
  case R600::CEIL:
  case R600::COS_cm:
  case R600::COS_eg:
  case R600::COS_r600:
  case R600::COS_r700:
  case R600::EXP_IEEE_cm:
  case R600::EXP_IEEE_eg:
  case R600::EXP_IEEE_r600:
  case R600::FFBH_UINT:
  case R600::FFBL_INT:
  case R600::FLOOR:
  case R600::FLT16_TO_FLT32:
  case R600::FLT32_TO_FLT16:
  case R600::FLT_TO_INT_eg:
  case R600::FLT_TO_INT_r600:
  case R600::FLT_TO_UINT_eg:
  case R600::FLT_TO_UINT_r600:
  case R600::FRACT:
  case R600::INTERP_LOAD_P0:
  case R600::INT_TO_FLT_eg:
  case R600::INT_TO_FLT_r600:
  case R600::LOG_CLAMPED_eg:
  case R600::LOG_CLAMPED_r600:
  case R600::LOG_IEEE_cm:
  case R600::LOG_IEEE_eg:
  case R600::LOG_IEEE_r600:
  case R600::MOV:
  case R600::MOVA_INT_eg:
  case R600::NOT_INT:
  case R600::RECIPSQRT_CLAMPED_cm:
  case R600::RECIPSQRT_CLAMPED_eg:
  case R600::RECIPSQRT_CLAMPED_r600:
  case R600::RECIPSQRT_IEEE_cm:
  case R600::RECIPSQRT_IEEE_eg:
  case R600::RECIPSQRT_IEEE_r600:
  case R600::RECIP_CLAMPED_cm:
  case R600::RECIP_CLAMPED_eg:
  case R600::RECIP_CLAMPED_r600:
  case R600::RECIP_IEEE_cm:
  case R600::RECIP_IEEE_eg:
  case R600::RECIP_IEEE_r600:
  case R600::RECIP_UINT_eg:
  case R600::RECIP_UINT_r600:
  case R600::RNDNE:
  case R600::SIN_cm:
  case R600::SIN_eg:
  case R600::SIN_r600:
  case R600::SIN_r700:
  case R600::TRUNC:
  case R600::UINT_TO_FLT_eg:
  case R600::UINT_TO_FLT_r600:
    return OperandMap[5][NamedIdx];
  case R600::ADD:
  case R600::ADDC_UINT:
  case R600::ADD_INT:
  case R600::AND_INT:
  case R600::ASHR_eg:
  case R600::ASHR_r600:
  case R600::BFM_INT_eg:
  case R600::CUBE_eg_real:
  case R600::CUBE_r600_real:
  case R600::DOT4_eg:
  case R600::DOT4_r600:
  case R600::INTERP_XY:
  case R600::INTERP_ZW:
  case R600::KILLGT:
  case R600::LSHL_eg:
  case R600::LSHL_r600:
  case R600::LSHR_eg:
  case R600::LSHR_r600:
  case R600::MAX:
  case R600::MAX_DX10:
  case R600::MAX_INT:
  case R600::MAX_UINT:
  case R600::MIN:
  case R600::MIN_DX10:
  case R600::MIN_INT:
  case R600::MIN_UINT:
  case R600::MUL:
  case R600::MULHI_INT_cm:
  case R600::MULHI_INT_cm24:
  case R600::MULHI_INT_eg:
  case R600::MULHI_INT_r600:
  case R600::MULHI_UINT24_eg:
  case R600::MULHI_UINT_cm:
  case R600::MULHI_UINT_cm24:
  case R600::MULHI_UINT_eg:
  case R600::MULHI_UINT_r600:
  case R600::MULLO_INT_cm:
  case R600::MULLO_INT_eg:
  case R600::MULLO_INT_r600:
  case R600::MULLO_UINT_cm:
  case R600::MULLO_UINT_eg:
  case R600::MULLO_UINT_r600:
  case R600::MUL_IEEE:
  case R600::MUL_INT24_cm:
  case R600::MUL_UINT24_eg:
  case R600::OR_INT:
  case R600::PRED_SETE:
  case R600::PRED_SETE_INT:
  case R600::PRED_SETGE:
  case R600::PRED_SETGE_INT:
  case R600::PRED_SETGT:
  case R600::PRED_SETGT_INT:
  case R600::PRED_SETNE:
  case R600::PRED_SETNE_INT:
  case R600::SETE:
  case R600::SETE_DX10:
  case R600::SETE_INT:
  case R600::SETGE_DX10:
  case R600::SETGE_INT:
  case R600::SETGE_UINT:
  case R600::SETGT_DX10:
  case R600::SETGT_INT:
  case R600::SETGT_UINT:
  case R600::SETNE_DX10:
  case R600::SETNE_INT:
  case R600::SGE:
  case R600::SGT:
  case R600::SNE:
  case R600::SUBB_UINT:
  case R600::SUB_INT:
  case R600::XOR_INT:
    return OperandMap[6][NamedIdx];
  case R600::DOT_4:
    return OperandMap[7][NamedIdx];
  case R600::R600_RegisterLoad:
    return OperandMap[8][NamedIdx];
  case R600::LDS_ADD:
  case R600::LDS_AND:
  case R600::LDS_BYTE_WRITE:
  case R600::LDS_MAX_INT:
  case R600::LDS_MAX_UINT:
  case R600::LDS_MIN_INT:
  case R600::LDS_MIN_UINT:
  case R600::LDS_OR:
  case R600::LDS_SHORT_WRITE:
  case R600::LDS_SUB:
  case R600::LDS_WRITE:
  case R600::LDS_WRXCHG:
  case R600::LDS_XOR:
    return OperandMap[9][NamedIdx];
  case R600::LDS_CMPST:
    return OperandMap[10][NamedIdx];
  case R600::R600_RegisterStore:
    return OperandMap[11][NamedIdx];
  case R600::CF_ALU:
  case R600::CF_ALU_BREAK:
  case R600::CF_ALU_CONTINUE:
  case R600::CF_ALU_ELSE_AFTER:
  case R600::CF_ALU_POP_AFTER:
  case R600::CF_ALU_PUSH_BEFORE:
    return OperandMap[12][NamedIdx];
    default: return -1;
  }
}
} // end namespace R600
} // end namespace llvm
#endif //GET_INSTRINFO_NAMED_OPS

#ifdef GET_INSTRINFO_OPERAND_TYPES_ENUM
#undef GET_INSTRINFO_OPERAND_TYPES_ENUM
namespace llvm {
namespace R600 {
namespace OpTypes {
enum OperandType {
  ABS = 0,
  BANK_SWIZZLE = 1,
  CLAMP = 2,
  CT = 3,
  FRAMEri = 4,
  InstFlag = 5,
  KCACHE = 6,
  LAST = 7,
  LITERAL = 8,
  MEMrr = 9,
  MEMxi = 10,
  NEG = 11,
  OMOD = 12,
  R600_Pred = 13,
  REL = 14,
  RSel = 15,
  SEL = 16,
  UEM = 17,
  UP = 18,
  WRITE = 19,
  brtarget = 20,
  f32imm = 21,
  f64imm = 22,
  i16imm = 23,
  i1imm = 24,
  i32imm = 25,
  i64imm = 26,
  i8imm = 27,
  ptype0 = 28,
  ptype1 = 29,
  ptype2 = 30,
  ptype3 = 31,
  ptype4 = 32,
  ptype5 = 33,
  s16imm = 34,
  type0 = 35,
  type1 = 36,
  type2 = 37,
  type3 = 38,
  type4 = 39,
  type5 = 40,
  u16imm = 41,
  u32imm = 42,
  u8imm = 43,
  untyped_imm_0 = 44,
  R600_Addr = 45,
  R600_Addr_W = 46,
  R600_Addr_Y = 47,
  R600_Addr_Z = 48,
  R600_ArrayBase = 49,
  R600_KC0 = 50,
  R600_KC0_W = 51,
  R600_KC0_X = 52,
  R600_KC0_Y = 53,
  R600_KC0_Z = 54,
  R600_KC1 = 55,
  R600_KC1_W = 56,
  R600_KC1_X = 57,
  R600_KC1_Y = 58,
  R600_KC1_Z = 59,
  R600_LDS_SRC_REG = 60,
  R600_Predicate = 61,
  R600_Predicate_Bit = 62,
  R600_Reg128 = 63,
  R600_Reg128Vertical = 64,
  R600_Reg32 = 65,
  R600_Reg64 = 66,
  R600_Reg64Vertical = 67,
  R600_TReg32 = 68,
  R600_TReg32_W = 69,
  R600_TReg32_X = 70,
  R600_TReg32_Y = 71,
  R600_TReg32_Z = 72,
  OPERAND_TYPE_LIST_END
};
} // end namespace OpTypes
} // end namespace R600
} // end namespace llvm
#endif // GET_INSTRINFO_OPERAND_TYPES_ENUM

#ifdef GET_INSTRINFO_OPERAND_TYPE
#undef GET_INSTRINFO_OPERAND_TYPE
namespace llvm {
namespace R600 {
LLVM_READONLY
static int getOperandType(uint16_t Opcode, uint16_t OpIdx) {
  const int Offsets[] = {
    0,
    1,
    1,
    1,
    2,
    3,
    4,
    5,
    5,
    8,
    12,
    13,
    17,
    20,
    20,
    20,
    21,
    23,
    25,
    25,
    26,
    27,
    29,
    29,
    35,
    36,
    37,
    40,
    40,
    42,
    43,
    43,
    43,
    43,
    43,
    43,
    45,
    48,
    48,
    51,
    54,
    57,
    60,
    63,
    66,
    69,
    72,
    75,
    78,
    79,
    80,
    82,
    84,
    87,
    89,
    93,
    95,
    97,
    99,
    101,
    103,
    105,
    107,
    109,
    111,
    113,
    115,
    117,
    118,
    120,
    122,
    124,
    129,
    134,
    139,
    141,
    146,
    151,
    155,
    158,
    161,
    164,
    167,
    170,
    173,
    176,
    179,
    182,
    185,
    188,
    191,
    194,
    196,
    198,
    199,
    200,
    201,
    203,
    205,
    207,
    209,
    210,
    213,
    215,
    218,
    220,
    223,
    226,
    229,
    233,
    237,
    241,
    245,
    249,
    253,
    258,
    262,
    267,
    271,
    276,
    280,
    285,
    289,
    293,
    296,
    299,
    302,
    305,
    308,
    311,
    314,
    317,
    321,
    325,
    329,
    333,
    337,
    341,
    345,
    349,
    352,
    355,
    358,
    362,
    366,
    369,
    372,
    375,
    378,
    380,
    382,
    384,
    386,
    388,
    390,
    392,
    394,
    396,
    398,
    400,
    402,
    404,
    407,
    409,
    412,
    415,
    418,
    421,
    424,
    427,
    430,
    433,
    436,
    439,
    442,
    445,
    447,
    448,
    451,
    455,
    458,
    462,
    464,
    466,
    468,
    470,
    472,
    474,
    476,
    478,
    480,
    482,
    484,
    486,
    488,
    490,
    492,
    494,
    496,
    499,
    502,
    505,
    508,
    511,
    514,
    518,
    520,
    522,
    524,
    528,
    532,
    536,
    539,
    542,
    544,
    546,
    548,
    550,
    552,
    554,
    556,
    558,
    560,
    562,
    564,
    566,
    568,
    569,
    571,
    573,
    573,
    575,
    577,
    578,
    579,
    580,
    581,
    583,
    583,
    585,
    587,
    588,
    589,
    590,
    591,
    593,
    595,
    595,
    666,
    666,
    666,
    666,
    666,
    666,
    666,
    666,
    666,
    668,
    670,
    670,
    672,
    674,
    675,
    676,
    677,
    678,
    679,
    680,
    682,
    683,
    685,
    687,
    689,
    693,
    696,
    699,
    703,
    707,
    711,
    715,
    715,
    715,
    722,
    729,
    729,
    750,
    771,
    792,
    793,
    814,
    835,
    856,
    870,
    889,
    908,
    927,
    948,
    967,
    981,
    990,
    999,
    1008,
    1017,
    1026,
    1035,
    1035,
    1035,
    1036,
    1037,
    1039,
    1041,
    1041,
    1041,
    1041,
    1043,
    1045,
    1047,
    1048,
    1050,
    1052,
    1054,
    1056,
    1075,
    1094,
    1113,
    1132,
    1151,
    1170,
    1189,
    1208,
    1227,
    1241,
    1255,
    1269,
    1283,
    1304,
    1325,
    1346,
    1367,
    1374,
    1383,
    1384,
    1385,
    1399,
    1413,
    1427,
    1428,
    1442,
    1456,
    1470,
    1484,
    1498,
    1512,
    1526,
    1540,
    1554,
    1573,
    1587,
    1587,
    1601,
    1606,
    1611,
    1613,
    1634,
    1655,
    1669,
    1683,
    1704,
    1713,
    1723,
    1732,
    1742,
    1749,
    1758,
    1770,
    1783,
    1792,
    1802,
    1811,
    1821,
    1830,
    1840,
    1849,
    1859,
    1868,
    1878,
    1885,
    1892,
    1901,
    1910,
    1920,
    1927,
    1934,
    1943,
    1952,
    1962,
    1971,
    1981,
    1983,
    1997,
    2011,
    2025,
    2039,
    2053,
    2054,
    2055,
    2076,
    2097,
    2118,
    2139,
    2160,
    2181,
    2202,
    2223,
    2244,
    2265,
    2286,
    2307,
    2321,
    2335,
    2356,
    2375,
    2394,
    2413,
    2432,
    2451,
    2470,
    2491,
    2512,
    2533,
    2554,
    2575,
    2596,
    2617,
    2638,
    2659,
    2680,
    2701,
    2722,
    2743,
    2764,
    2785,
    2806,
    2827,
    2846,
    2865,
    2886,
    2900,
    2921,
    2921,
    2923,
    2925,
    2946,
    2967,
    2988,
    3009,
    3030,
    3051,
    3072,
    3093,
    3100,
    3109,
    3112,
    3115,
    3118,
    3121,
    3124,
    3127,
    3130,
    3133,
    3136,
    3139,
    3142,
    3145,
    3148,
    3151,
    3154,
    3157,
    3160,
    3163,
    3166,
    3169,
    3172,
    3175,
    3178,
    3181,
    3184,
    3187,
    3190,
    3193,
    3195,
    3197,
    3199,
    3201,
    3205,
    3209,
    3212,
    3215,
    3218,
    3232,
    3246,
    3260,
    3274,
    3288,
    3302,
    3316,
    3330,
    3344,
    3358,
    3372,
    3386,
    3400,
    3414,
    3428,
    3449,
    3470,
    3491,
    3512,
    3533,
    3554,
    3575,
    3596,
    3617,
    3638,
    3659,
    3680,
    3701,
    3715,
    3729,
    3743,
    3757,
    3778,
    3799,
    3820,
    3839,
    3858,
    3877,
    3896,
    3915,
    3934,
    3953,
    3972,
    3991,
    4010,
    4029,
    4048,
    4067,
    4086,
    4105,
    4109,
    4113,
    4127,
    4141,
    4155,
    4159,
    4163,
    4167,
    4171,
    4175,
    4179,
    4183,
    4187,
    4191,
    4195,
    4196,
    4197,
  };
  const int OpcodeOperandTypes[] = {
    -1, 
    /**/
    /**/
    OpTypes::i32imm, 
    OpTypes::i32imm, 
    OpTypes::i32imm, 
    OpTypes::i32imm, 
    /**/
    -1, -1, OpTypes::i32imm, 
    -1, -1, -1, OpTypes::i32imm, 
    -1, 
    -1, -1, -1, OpTypes::i32imm, 
    -1, -1, OpTypes::i32imm, 
    /**/
    /**/
    -1, 
    -1, -1, 
    -1, -1, 
    /**/
    OpTypes::i32imm, 
    OpTypes::i32imm, 
    OpTypes::i64imm, OpTypes::i32imm, 
    /**/
    -1, OpTypes::i64imm, OpTypes::i32imm, -1, OpTypes::i32imm, OpTypes::i32imm, 
    -1, 
    OpTypes::i32imm, 
    -1, OpTypes::i32imm, OpTypes::i32imm, 
    /**/
    -1, OpTypes::i32imm, 
    -1, 
    /**/
    /**/
    /**/
    /**/
    /**/
    -1, -1, 
    -1, -1, -1, 
    /**/
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, 
    OpTypes::type0, 
    OpTypes::type0, -1, 
    OpTypes::type0, -1, 
    OpTypes::type0, OpTypes::type1, OpTypes::untyped_imm_0, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type0, OpTypes::type1, OpTypes::untyped_imm_0, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, 
    OpTypes::type0, OpTypes::ptype1, 
    OpTypes::type0, OpTypes::ptype1, 
    OpTypes::type0, OpTypes::ptype1, 
    OpTypes::type0, OpTypes::ptype1, OpTypes::ptype1, OpTypes::type2, -1, 
    OpTypes::type0, OpTypes::ptype1, OpTypes::ptype1, OpTypes::type2, -1, 
    OpTypes::type0, OpTypes::ptype1, OpTypes::ptype1, OpTypes::type2, -1, 
    OpTypes::type0, OpTypes::ptype1, 
    OpTypes::ptype0, OpTypes::type1, OpTypes::ptype0, OpTypes::ptype2, -1, 
    OpTypes::type0, OpTypes::type1, OpTypes::type2, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::ptype1, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::ptype1, OpTypes::type0, 
    OpTypes::type0, OpTypes::ptype1, OpTypes::type0, 
    OpTypes::type0, OpTypes::ptype1, OpTypes::type0, 
    OpTypes::type0, OpTypes::ptype1, OpTypes::type0, 
    OpTypes::type0, OpTypes::ptype1, OpTypes::type0, 
    OpTypes::type0, OpTypes::ptype1, OpTypes::type0, 
    OpTypes::type0, OpTypes::ptype1, OpTypes::type0, 
    OpTypes::type0, OpTypes::ptype1, OpTypes::type0, 
    OpTypes::type0, OpTypes::ptype1, OpTypes::type0, 
    OpTypes::type0, OpTypes::ptype1, OpTypes::type0, 
    OpTypes::type0, OpTypes::ptype1, OpTypes::type0, 
    OpTypes::type0, OpTypes::ptype1, OpTypes::type0, 
    OpTypes::type0, OpTypes::ptype1, OpTypes::type0, 
    OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::type0, -1, 
    OpTypes::type0, 
    -1, 
    -1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, -1, 
    OpTypes::type0, -1, 
    OpTypes::type0, 
    OpTypes::type0, OpTypes::type1, -1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type0, OpTypes::untyped_imm_0, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, -1, OpTypes::type1, OpTypes::type1, 
    OpTypes::type0, -1, OpTypes::type1, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type1, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type1, OpTypes::type0, OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type1, OpTypes::type0, OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type1, OpTypes::type0, OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type1, OpTypes::type0, OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type1, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, OpTypes::untyped_imm_0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, OpTypes::untyped_imm_0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, OpTypes::untyped_imm_0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, OpTypes::untyped_imm_0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, OpTypes::untyped_imm_0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, OpTypes::untyped_imm_0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, OpTypes::untyped_imm_0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, OpTypes::untyped_imm_0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type1, 
    OpTypes::ptype0, OpTypes::ptype0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, 
    -1, 
    OpTypes::ptype0, -1, OpTypes::type1, 
    OpTypes::type0, OpTypes::type0, OpTypes::type1, OpTypes::type2, 
    OpTypes::type0, OpTypes::type1, OpTypes::type2, 
    OpTypes::type0, OpTypes::type1, OpTypes::type1, -1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, -1, 
    OpTypes::type0, -1, 
    OpTypes::ptype0, OpTypes::type1, OpTypes::i32imm, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, OpTypes::type0, 
    OpTypes::type0, -1, 
    -1, OpTypes::type0, 
    OpTypes::ptype0, OpTypes::ptype1, OpTypes::type2, OpTypes::untyped_imm_0, 
    OpTypes::ptype0, OpTypes::ptype1, OpTypes::type2, OpTypes::untyped_imm_0, 
    OpTypes::ptype0, OpTypes::type1, OpTypes::type2, OpTypes::untyped_imm_0, 
    OpTypes::type0, OpTypes::type1, OpTypes::type2, 
    OpTypes::type0, OpTypes::type1, OpTypes::type2, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::type0, OpTypes::type1, 
    OpTypes::brtarget, 
    OpTypes::brtarget, OpTypes::R600_Reg32, 
    OpTypes::brtarget, OpTypes::R600_Reg32, 
    /**/
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, 
    OpTypes::R600_Reg32, 
    OpTypes::R600_Reg32, 
    OpTypes::R600_Reg32, 
    OpTypes::R600_Reg32, 
    OpTypes::R600_Reg32, OpTypes::i32imm, 
    /**/
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, 
    OpTypes::R600_Reg32, 
    OpTypes::R600_Reg32, 
    OpTypes::R600_Reg32, 
    OpTypes::R600_Reg32, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, 
    /**/
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_TReg32_X, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_TReg32_X, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Predicate, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_TReg32_Y, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_TReg32_Y, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Predicate, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_TReg32_Z, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_TReg32_Z, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Predicate, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_TReg32_W, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_TReg32_W, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::LITERAL, 
    /**/
    /**/
    /**/
    /**/
    /**/
    /**/
    /**/
    /**/
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, 
    /**/
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, 
    OpTypes::R600_Reg32, 
    OpTypes::R600_Reg32, 
    OpTypes::R600_Reg32, 
    OpTypes::R600_Reg32, 
    OpTypes::R600_Reg32, 
    OpTypes::brtarget, 
    OpTypes::brtarget, OpTypes::R600_Predicate_Bit, 
    OpTypes::R600_Reg32, 
    OpTypes::R600_Reg32, OpTypes::f32imm, 
    OpTypes::R600_Reg32, OpTypes::i32imm, 
    OpTypes::R600_Reg32, OpTypes::i32imm, 
    OpTypes::R600_Predicate_Bit, OpTypes::R600_Reg32, OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg64Vertical, OpTypes::R600_Reg32, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg128Vertical, OpTypes::R600_Reg32, 
    OpTypes::R600_Reg64Vertical, OpTypes::R600_Reg64Vertical, OpTypes::R600_Reg32, OpTypes::R600_Reg32, 
    OpTypes::R600_Reg128Vertical, OpTypes::R600_Reg128Vertical, OpTypes::R600_Reg32, OpTypes::R600_Reg32, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, OpTypes::i32imm, OpTypes::i32imm, 
    /**/
    /**/
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, 
    /**/
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::i32imm, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::KCACHE, OpTypes::KCACHE, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::KCACHE, OpTypes::KCACHE, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::KCACHE, OpTypes::KCACHE, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::KCACHE, OpTypes::KCACHE, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::KCACHE, OpTypes::KCACHE, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::KCACHE, OpTypes::KCACHE, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, 
    /**/
    /**/
    OpTypes::i32imm, 
    OpTypes::i32imm, 
    OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::i32imm, OpTypes::i32imm, 
    /**/
    /**/
    /**/
    OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::i32imm, 
    OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg128, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::R600_Reg128, OpTypes::i32imm, OpTypes::i32imm, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::i32imm, 
    OpTypes::i32imm, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::i32imm, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    /**/
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_TReg32_X, OpTypes::R600_TReg32_Y, OpTypes::i32imm, OpTypes::R600_TReg32_Y, OpTypes::R600_TReg32_X, 
    OpTypes::R600_TReg32_Z, OpTypes::R600_TReg32_W, OpTypes::i32imm, OpTypes::R600_TReg32_Y, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::i32imm, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::BANK_SWIZZLE, 
    OpTypes::LITERAL, OpTypes::LITERAL, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::i32imm, 
    OpTypes::i32imm, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    /**/
    OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg128, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::R600_Reg128, OpTypes::i32imm, OpTypes::i32imm, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, 
    OpTypes::R600_TReg32_X, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg64, OpTypes::R600_TReg32_X, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::i32imm, OpTypes::InstFlag, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::i32imm, OpTypes::InstFlag, 
    OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, OpTypes::InstFlag, 
    OpTypes::R600_TReg32_X, OpTypes::R600_TReg32_X, OpTypes::InstFlag, 
    OpTypes::R600_Reg64, OpTypes::R600_TReg32_X, OpTypes::InstFlag, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::CT, OpTypes::CT, OpTypes::CT, OpTypes::CT, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::CT, OpTypes::CT, OpTypes::CT, OpTypes::CT, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::CT, OpTypes::CT, OpTypes::CT, OpTypes::CT, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::CT, OpTypes::CT, OpTypes::CT, OpTypes::CT, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::CT, OpTypes::CT, OpTypes::CT, OpTypes::CT, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::CT, OpTypes::CT, OpTypes::CT, OpTypes::CT, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::CT, OpTypes::CT, OpTypes::CT, OpTypes::CT, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::CT, OpTypes::CT, OpTypes::CT, OpTypes::CT, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::CT, OpTypes::CT, OpTypes::CT, OpTypes::CT, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::CT, OpTypes::CT, OpTypes::CT, OpTypes::CT, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::CT, OpTypes::CT, OpTypes::CT, OpTypes::CT, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::CT, OpTypes::CT, OpTypes::CT, OpTypes::CT, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::CT, OpTypes::CT, OpTypes::CT, OpTypes::CT, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::CT, OpTypes::CT, OpTypes::CT, OpTypes::CT, 
    OpTypes::R600_Reg128, OpTypes::R600_Reg128, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::i32imm, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::RSel, OpTypes::i32imm, OpTypes::i32imm, OpTypes::CT, OpTypes::CT, OpTypes::CT, OpTypes::CT, 
    OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, OpTypes::i32imm, OpTypes::i32imm, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg32, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
    OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, OpTypes::i32imm, OpTypes::i8imm, 
    OpTypes::R600_Reg128, OpTypes::R600_TReg32_X, OpTypes::i32imm, OpTypes::i8imm, 
    OpTypes::R600_TReg32_X, OpTypes::R600_TReg32_X, OpTypes::i32imm, OpTypes::i8imm, 
    OpTypes::R600_TReg32_X, OpTypes::R600_TReg32_X, OpTypes::i32imm, OpTypes::i8imm, 
    OpTypes::R600_TReg32_X, OpTypes::R600_TReg32_X, OpTypes::i32imm, OpTypes::i8imm, 
    OpTypes::R600_TReg32_X, OpTypes::R600_TReg32_X, OpTypes::i32imm, OpTypes::i8imm, 
    OpTypes::R600_Reg64, OpTypes::R600_TReg32_X, OpTypes::i32imm, OpTypes::i8imm, 
    OpTypes::R600_Reg64, OpTypes::R600_TReg32_X, OpTypes::i32imm, OpTypes::i8imm, 
    OpTypes::R600_TReg32_X, OpTypes::R600_TReg32_X, OpTypes::i32imm, OpTypes::i8imm, 
    OpTypes::R600_TReg32_X, OpTypes::R600_TReg32_X, OpTypes::i32imm, OpTypes::i8imm, 
    OpTypes::i32imm, 
    OpTypes::i32imm, 
    OpTypes::R600_Reg32, OpTypes::UEM, OpTypes::UP, OpTypes::WRITE, OpTypes::OMOD, OpTypes::REL, OpTypes::CLAMP, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::R600_Reg32, OpTypes::NEG, OpTypes::REL, OpTypes::ABS, OpTypes::SEL, OpTypes::LAST, OpTypes::R600_Predicate, OpTypes::LITERAL, OpTypes::BANK_SWIZZLE, 
  };
  return OpcodeOperandTypes[Offsets[Opcode] + OpIdx];
}
} // end namespace R600
} // end namespace llvm
#endif // GET_INSTRINFO_OPERAND_TYPE

#ifdef GET_INSTRMAP_INFO
#undef GET_INSTRMAP_INFO
namespace llvm {

namespace R600 {

enum DisableEncoding {
	DisableEncoding_
};

// getLDSNoRetOp
LLVM_READONLY
int getLDSNoRetOp(uint16_t Opcode) {
static const uint16_t getLDSNoRetOpTable[][2] = {
  { R600::LDS_ADD_RET, R600::LDS_ADD },
  { R600::LDS_AND_RET, R600::LDS_AND },
  { R600::LDS_MAX_INT_RET, R600::LDS_MAX_INT },
  { R600::LDS_MAX_UINT_RET, R600::LDS_MAX_UINT },
  { R600::LDS_MIN_INT_RET, R600::LDS_MIN_INT },
  { R600::LDS_MIN_UINT_RET, R600::LDS_MIN_UINT },
  { R600::LDS_OR_RET, R600::LDS_OR },
  { R600::LDS_SUB_RET, R600::LDS_SUB },
  { R600::LDS_WRXCHG_RET, R600::LDS_WRXCHG },
  { R600::LDS_XOR_RET, R600::LDS_XOR },
}; // End of getLDSNoRetOpTable

  unsigned mid;
  unsigned start = 0;
  unsigned end = 10;
  while (start < end) {
    mid = start + (end - start)/2;
    if (Opcode == getLDSNoRetOpTable[mid][0]) {
      break;
    }
    if (Opcode < getLDSNoRetOpTable[mid][0])
      end = mid;
    else
      start = mid + 1;
  }
  if (start == end)
    return -1; // Instruction doesn't exist in this table.

  return getLDSNoRetOpTable[mid][1];
}

} // end namespace R600
} // end namespace llvm
#endif // GET_INSTRMAP_INFO

