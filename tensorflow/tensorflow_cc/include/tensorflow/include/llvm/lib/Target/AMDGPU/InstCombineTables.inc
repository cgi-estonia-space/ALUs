#ifdef GET_AMDGPUImageDMaskIntrinsicTable_DECL
const AMDGPUImageDMaskIntrinsic *getAMDGPUImageDMaskIntrinsic(unsigned Intr);
#endif

#ifdef GET_AMDGPUImageDMaskIntrinsicTable_IMPL
constexpr AMDGPUImageDMaskIntrinsic AMDGPUImageDMaskIntrinsicTable[] = {
  { Intrinsic::amdgcn_image_getlod_1d }, // 0
  { Intrinsic::amdgcn_image_getlod_1darray }, // 1
  { Intrinsic::amdgcn_image_getlod_2d }, // 2
  { Intrinsic::amdgcn_image_getlod_2darray }, // 3
  { Intrinsic::amdgcn_image_getlod_3d }, // 4
  { Intrinsic::amdgcn_image_getlod_cube }, // 5
  { Intrinsic::amdgcn_image_getresinfo_1d }, // 6
  { Intrinsic::amdgcn_image_getresinfo_1darray }, // 7
  { Intrinsic::amdgcn_image_getresinfo_2d }, // 8
  { Intrinsic::amdgcn_image_getresinfo_2darray }, // 9
  { Intrinsic::amdgcn_image_getresinfo_2darraymsaa }, // 10
  { Intrinsic::amdgcn_image_getresinfo_2dmsaa }, // 11
  { Intrinsic::amdgcn_image_getresinfo_3d }, // 12
  { Intrinsic::amdgcn_image_getresinfo_cube }, // 13
  { Intrinsic::amdgcn_image_load_1d }, // 14
  { Intrinsic::amdgcn_image_load_1darray }, // 15
  { Intrinsic::amdgcn_image_load_2d }, // 16
  { Intrinsic::amdgcn_image_load_2darray }, // 17
  { Intrinsic::amdgcn_image_load_2darraymsaa }, // 18
  { Intrinsic::amdgcn_image_load_2dmsaa }, // 19
  { Intrinsic::amdgcn_image_load_3d }, // 20
  { Intrinsic::amdgcn_image_load_cube }, // 21
  { Intrinsic::amdgcn_image_load_mip_1d }, // 22
  { Intrinsic::amdgcn_image_load_mip_1darray }, // 23
  { Intrinsic::amdgcn_image_load_mip_2d }, // 24
  { Intrinsic::amdgcn_image_load_mip_2darray }, // 25
  { Intrinsic::amdgcn_image_load_mip_3d }, // 26
  { Intrinsic::amdgcn_image_load_mip_cube }, // 27
  { Intrinsic::amdgcn_image_msaa_load_1d }, // 28
  { Intrinsic::amdgcn_image_msaa_load_1darray }, // 29
  { Intrinsic::amdgcn_image_msaa_load_2d }, // 30
  { Intrinsic::amdgcn_image_msaa_load_2darray }, // 31
  { Intrinsic::amdgcn_image_msaa_load_2darraymsaa }, // 32
  { Intrinsic::amdgcn_image_msaa_load_2dmsaa }, // 33
  { Intrinsic::amdgcn_image_msaa_load_3d }, // 34
  { Intrinsic::amdgcn_image_msaa_load_cube }, // 35
  { Intrinsic::amdgcn_image_sample_1d }, // 36
  { Intrinsic::amdgcn_image_sample_1darray }, // 37
  { Intrinsic::amdgcn_image_sample_2d }, // 38
  { Intrinsic::amdgcn_image_sample_2darray }, // 39
  { Intrinsic::amdgcn_image_sample_3d }, // 40
  { Intrinsic::amdgcn_image_sample_b_1d }, // 41
  { Intrinsic::amdgcn_image_sample_b_1darray }, // 42
  { Intrinsic::amdgcn_image_sample_b_2d }, // 43
  { Intrinsic::amdgcn_image_sample_b_2darray }, // 44
  { Intrinsic::amdgcn_image_sample_b_3d }, // 45
  { Intrinsic::amdgcn_image_sample_b_cl_1d }, // 46
  { Intrinsic::amdgcn_image_sample_b_cl_1darray }, // 47
  { Intrinsic::amdgcn_image_sample_b_cl_2d }, // 48
  { Intrinsic::amdgcn_image_sample_b_cl_2darray }, // 49
  { Intrinsic::amdgcn_image_sample_b_cl_3d }, // 50
  { Intrinsic::amdgcn_image_sample_b_cl_cube }, // 51
  { Intrinsic::amdgcn_image_sample_b_cl_o_1d }, // 52
  { Intrinsic::amdgcn_image_sample_b_cl_o_1darray }, // 53
  { Intrinsic::amdgcn_image_sample_b_cl_o_2d }, // 54
  { Intrinsic::amdgcn_image_sample_b_cl_o_2darray }, // 55
  { Intrinsic::amdgcn_image_sample_b_cl_o_3d }, // 56
  { Intrinsic::amdgcn_image_sample_b_cl_o_cube }, // 57
  { Intrinsic::amdgcn_image_sample_b_cube }, // 58
  { Intrinsic::amdgcn_image_sample_b_o_1d }, // 59
  { Intrinsic::amdgcn_image_sample_b_o_1darray }, // 60
  { Intrinsic::amdgcn_image_sample_b_o_2d }, // 61
  { Intrinsic::amdgcn_image_sample_b_o_2darray }, // 62
  { Intrinsic::amdgcn_image_sample_b_o_3d }, // 63
  { Intrinsic::amdgcn_image_sample_b_o_cube }, // 64
  { Intrinsic::amdgcn_image_sample_c_1d }, // 65
  { Intrinsic::amdgcn_image_sample_c_1darray }, // 66
  { Intrinsic::amdgcn_image_sample_c_2d }, // 67
  { Intrinsic::amdgcn_image_sample_c_2darray }, // 68
  { Intrinsic::amdgcn_image_sample_c_3d }, // 69
  { Intrinsic::amdgcn_image_sample_c_b_1d }, // 70
  { Intrinsic::amdgcn_image_sample_c_b_1darray }, // 71
  { Intrinsic::amdgcn_image_sample_c_b_2d }, // 72
  { Intrinsic::amdgcn_image_sample_c_b_2darray }, // 73
  { Intrinsic::amdgcn_image_sample_c_b_3d }, // 74
  { Intrinsic::amdgcn_image_sample_c_b_cl_1d }, // 75
  { Intrinsic::amdgcn_image_sample_c_b_cl_1darray }, // 76
  { Intrinsic::amdgcn_image_sample_c_b_cl_2d }, // 77
  { Intrinsic::amdgcn_image_sample_c_b_cl_2darray }, // 78
  { Intrinsic::amdgcn_image_sample_c_b_cl_3d }, // 79
  { Intrinsic::amdgcn_image_sample_c_b_cl_cube }, // 80
  { Intrinsic::amdgcn_image_sample_c_b_cl_o_1d }, // 81
  { Intrinsic::amdgcn_image_sample_c_b_cl_o_1darray }, // 82
  { Intrinsic::amdgcn_image_sample_c_b_cl_o_2d }, // 83
  { Intrinsic::amdgcn_image_sample_c_b_cl_o_2darray }, // 84
  { Intrinsic::amdgcn_image_sample_c_b_cl_o_3d }, // 85
  { Intrinsic::amdgcn_image_sample_c_b_cl_o_cube }, // 86
  { Intrinsic::amdgcn_image_sample_c_b_cube }, // 87
  { Intrinsic::amdgcn_image_sample_c_b_o_1d }, // 88
  { Intrinsic::amdgcn_image_sample_c_b_o_1darray }, // 89
  { Intrinsic::amdgcn_image_sample_c_b_o_2d }, // 90
  { Intrinsic::amdgcn_image_sample_c_b_o_2darray }, // 91
  { Intrinsic::amdgcn_image_sample_c_b_o_3d }, // 92
  { Intrinsic::amdgcn_image_sample_c_b_o_cube }, // 93
  { Intrinsic::amdgcn_image_sample_c_cd_1d }, // 94
  { Intrinsic::amdgcn_image_sample_c_cd_1darray }, // 95
  { Intrinsic::amdgcn_image_sample_c_cd_2d }, // 96
  { Intrinsic::amdgcn_image_sample_c_cd_2darray }, // 97
  { Intrinsic::amdgcn_image_sample_c_cd_3d }, // 98
  { Intrinsic::amdgcn_image_sample_c_cd_cl_1d }, // 99
  { Intrinsic::amdgcn_image_sample_c_cd_cl_1darray }, // 100
  { Intrinsic::amdgcn_image_sample_c_cd_cl_2d }, // 101
  { Intrinsic::amdgcn_image_sample_c_cd_cl_2darray }, // 102
  { Intrinsic::amdgcn_image_sample_c_cd_cl_3d }, // 103
  { Intrinsic::amdgcn_image_sample_c_cd_cl_cube }, // 104
  { Intrinsic::amdgcn_image_sample_c_cd_cl_o_1d }, // 105
  { Intrinsic::amdgcn_image_sample_c_cd_cl_o_1darray }, // 106
  { Intrinsic::amdgcn_image_sample_c_cd_cl_o_2d }, // 107
  { Intrinsic::amdgcn_image_sample_c_cd_cl_o_2darray }, // 108
  { Intrinsic::amdgcn_image_sample_c_cd_cl_o_3d }, // 109
  { Intrinsic::amdgcn_image_sample_c_cd_cl_o_cube }, // 110
  { Intrinsic::amdgcn_image_sample_c_cd_cube }, // 111
  { Intrinsic::amdgcn_image_sample_c_cd_o_1d }, // 112
  { Intrinsic::amdgcn_image_sample_c_cd_o_1darray }, // 113
  { Intrinsic::amdgcn_image_sample_c_cd_o_2d }, // 114
  { Intrinsic::amdgcn_image_sample_c_cd_o_2darray }, // 115
  { Intrinsic::amdgcn_image_sample_c_cd_o_3d }, // 116
  { Intrinsic::amdgcn_image_sample_c_cd_o_cube }, // 117
  { Intrinsic::amdgcn_image_sample_c_cl_1d }, // 118
  { Intrinsic::amdgcn_image_sample_c_cl_1darray }, // 119
  { Intrinsic::amdgcn_image_sample_c_cl_2d }, // 120
  { Intrinsic::amdgcn_image_sample_c_cl_2darray }, // 121
  { Intrinsic::amdgcn_image_sample_c_cl_3d }, // 122
  { Intrinsic::amdgcn_image_sample_c_cl_cube }, // 123
  { Intrinsic::amdgcn_image_sample_c_cl_o_1d }, // 124
  { Intrinsic::amdgcn_image_sample_c_cl_o_1darray }, // 125
  { Intrinsic::amdgcn_image_sample_c_cl_o_2d }, // 126
  { Intrinsic::amdgcn_image_sample_c_cl_o_2darray }, // 127
  { Intrinsic::amdgcn_image_sample_c_cl_o_3d }, // 128
  { Intrinsic::amdgcn_image_sample_c_cl_o_cube }, // 129
  { Intrinsic::amdgcn_image_sample_c_cube }, // 130
  { Intrinsic::amdgcn_image_sample_c_d_1d }, // 131
  { Intrinsic::amdgcn_image_sample_c_d_1darray }, // 132
  { Intrinsic::amdgcn_image_sample_c_d_2d }, // 133
  { Intrinsic::amdgcn_image_sample_c_d_2darray }, // 134
  { Intrinsic::amdgcn_image_sample_c_d_3d }, // 135
  { Intrinsic::amdgcn_image_sample_c_d_cl_1d }, // 136
  { Intrinsic::amdgcn_image_sample_c_d_cl_1darray }, // 137
  { Intrinsic::amdgcn_image_sample_c_d_cl_2d }, // 138
  { Intrinsic::amdgcn_image_sample_c_d_cl_2darray }, // 139
  { Intrinsic::amdgcn_image_sample_c_d_cl_3d }, // 140
  { Intrinsic::amdgcn_image_sample_c_d_cl_cube }, // 141
  { Intrinsic::amdgcn_image_sample_c_d_cl_o_1d }, // 142
  { Intrinsic::amdgcn_image_sample_c_d_cl_o_1darray }, // 143
  { Intrinsic::amdgcn_image_sample_c_d_cl_o_2d }, // 144
  { Intrinsic::amdgcn_image_sample_c_d_cl_o_2darray }, // 145
  { Intrinsic::amdgcn_image_sample_c_d_cl_o_3d }, // 146
  { Intrinsic::amdgcn_image_sample_c_d_cl_o_cube }, // 147
  { Intrinsic::amdgcn_image_sample_c_d_cube }, // 148
  { Intrinsic::amdgcn_image_sample_c_d_o_1d }, // 149
  { Intrinsic::amdgcn_image_sample_c_d_o_1darray }, // 150
  { Intrinsic::amdgcn_image_sample_c_d_o_2d }, // 151
  { Intrinsic::amdgcn_image_sample_c_d_o_2darray }, // 152
  { Intrinsic::amdgcn_image_sample_c_d_o_3d }, // 153
  { Intrinsic::amdgcn_image_sample_c_d_o_cube }, // 154
  { Intrinsic::amdgcn_image_sample_c_l_1d }, // 155
  { Intrinsic::amdgcn_image_sample_c_l_1darray }, // 156
  { Intrinsic::amdgcn_image_sample_c_l_2d }, // 157
  { Intrinsic::amdgcn_image_sample_c_l_2darray }, // 158
  { Intrinsic::amdgcn_image_sample_c_l_3d }, // 159
  { Intrinsic::amdgcn_image_sample_c_l_cube }, // 160
  { Intrinsic::amdgcn_image_sample_c_l_o_1d }, // 161
  { Intrinsic::amdgcn_image_sample_c_l_o_1darray }, // 162
  { Intrinsic::amdgcn_image_sample_c_l_o_2d }, // 163
  { Intrinsic::amdgcn_image_sample_c_l_o_2darray }, // 164
  { Intrinsic::amdgcn_image_sample_c_l_o_3d }, // 165
  { Intrinsic::amdgcn_image_sample_c_l_o_cube }, // 166
  { Intrinsic::amdgcn_image_sample_c_lz_1d }, // 167
  { Intrinsic::amdgcn_image_sample_c_lz_1darray }, // 168
  { Intrinsic::amdgcn_image_sample_c_lz_2d }, // 169
  { Intrinsic::amdgcn_image_sample_c_lz_2darray }, // 170
  { Intrinsic::amdgcn_image_sample_c_lz_3d }, // 171
  { Intrinsic::amdgcn_image_sample_c_lz_cube }, // 172
  { Intrinsic::amdgcn_image_sample_c_lz_o_1d }, // 173
  { Intrinsic::amdgcn_image_sample_c_lz_o_1darray }, // 174
  { Intrinsic::amdgcn_image_sample_c_lz_o_2d }, // 175
  { Intrinsic::amdgcn_image_sample_c_lz_o_2darray }, // 176
  { Intrinsic::amdgcn_image_sample_c_lz_o_3d }, // 177
  { Intrinsic::amdgcn_image_sample_c_lz_o_cube }, // 178
  { Intrinsic::amdgcn_image_sample_c_o_1d }, // 179
  { Intrinsic::amdgcn_image_sample_c_o_1darray }, // 180
  { Intrinsic::amdgcn_image_sample_c_o_2d }, // 181
  { Intrinsic::amdgcn_image_sample_c_o_2darray }, // 182
  { Intrinsic::amdgcn_image_sample_c_o_3d }, // 183
  { Intrinsic::amdgcn_image_sample_c_o_cube }, // 184
  { Intrinsic::amdgcn_image_sample_cd_1d }, // 185
  { Intrinsic::amdgcn_image_sample_cd_1darray }, // 186
  { Intrinsic::amdgcn_image_sample_cd_2d }, // 187
  { Intrinsic::amdgcn_image_sample_cd_2darray }, // 188
  { Intrinsic::amdgcn_image_sample_cd_3d }, // 189
  { Intrinsic::amdgcn_image_sample_cd_cl_1d }, // 190
  { Intrinsic::amdgcn_image_sample_cd_cl_1darray }, // 191
  { Intrinsic::amdgcn_image_sample_cd_cl_2d }, // 192
  { Intrinsic::amdgcn_image_sample_cd_cl_2darray }, // 193
  { Intrinsic::amdgcn_image_sample_cd_cl_3d }, // 194
  { Intrinsic::amdgcn_image_sample_cd_cl_cube }, // 195
  { Intrinsic::amdgcn_image_sample_cd_cl_o_1d }, // 196
  { Intrinsic::amdgcn_image_sample_cd_cl_o_1darray }, // 197
  { Intrinsic::amdgcn_image_sample_cd_cl_o_2d }, // 198
  { Intrinsic::amdgcn_image_sample_cd_cl_o_2darray }, // 199
  { Intrinsic::amdgcn_image_sample_cd_cl_o_3d }, // 200
  { Intrinsic::amdgcn_image_sample_cd_cl_o_cube }, // 201
  { Intrinsic::amdgcn_image_sample_cd_cube }, // 202
  { Intrinsic::amdgcn_image_sample_cd_o_1d }, // 203
  { Intrinsic::amdgcn_image_sample_cd_o_1darray }, // 204
  { Intrinsic::amdgcn_image_sample_cd_o_2d }, // 205
  { Intrinsic::amdgcn_image_sample_cd_o_2darray }, // 206
  { Intrinsic::amdgcn_image_sample_cd_o_3d }, // 207
  { Intrinsic::amdgcn_image_sample_cd_o_cube }, // 208
  { Intrinsic::amdgcn_image_sample_cl_1d }, // 209
  { Intrinsic::amdgcn_image_sample_cl_1darray }, // 210
  { Intrinsic::amdgcn_image_sample_cl_2d }, // 211
  { Intrinsic::amdgcn_image_sample_cl_2darray }, // 212
  { Intrinsic::amdgcn_image_sample_cl_3d }, // 213
  { Intrinsic::amdgcn_image_sample_cl_cube }, // 214
  { Intrinsic::amdgcn_image_sample_cl_o_1d }, // 215
  { Intrinsic::amdgcn_image_sample_cl_o_1darray }, // 216
  { Intrinsic::amdgcn_image_sample_cl_o_2d }, // 217
  { Intrinsic::amdgcn_image_sample_cl_o_2darray }, // 218
  { Intrinsic::amdgcn_image_sample_cl_o_3d }, // 219
  { Intrinsic::amdgcn_image_sample_cl_o_cube }, // 220
  { Intrinsic::amdgcn_image_sample_cube }, // 221
  { Intrinsic::amdgcn_image_sample_d_1d }, // 222
  { Intrinsic::amdgcn_image_sample_d_1darray }, // 223
  { Intrinsic::amdgcn_image_sample_d_2d }, // 224
  { Intrinsic::amdgcn_image_sample_d_2darray }, // 225
  { Intrinsic::amdgcn_image_sample_d_3d }, // 226
  { Intrinsic::amdgcn_image_sample_d_cl_1d }, // 227
  { Intrinsic::amdgcn_image_sample_d_cl_1darray }, // 228
  { Intrinsic::amdgcn_image_sample_d_cl_2d }, // 229
  { Intrinsic::amdgcn_image_sample_d_cl_2darray }, // 230
  { Intrinsic::amdgcn_image_sample_d_cl_3d }, // 231
  { Intrinsic::amdgcn_image_sample_d_cl_cube }, // 232
  { Intrinsic::amdgcn_image_sample_d_cl_o_1d }, // 233
  { Intrinsic::amdgcn_image_sample_d_cl_o_1darray }, // 234
  { Intrinsic::amdgcn_image_sample_d_cl_o_2d }, // 235
  { Intrinsic::amdgcn_image_sample_d_cl_o_2darray }, // 236
  { Intrinsic::amdgcn_image_sample_d_cl_o_3d }, // 237
  { Intrinsic::amdgcn_image_sample_d_cl_o_cube }, // 238
  { Intrinsic::amdgcn_image_sample_d_cube }, // 239
  { Intrinsic::amdgcn_image_sample_d_o_1d }, // 240
  { Intrinsic::amdgcn_image_sample_d_o_1darray }, // 241
  { Intrinsic::amdgcn_image_sample_d_o_2d }, // 242
  { Intrinsic::amdgcn_image_sample_d_o_2darray }, // 243
  { Intrinsic::amdgcn_image_sample_d_o_3d }, // 244
  { Intrinsic::amdgcn_image_sample_d_o_cube }, // 245
  { Intrinsic::amdgcn_image_sample_l_1d }, // 246
  { Intrinsic::amdgcn_image_sample_l_1darray }, // 247
  { Intrinsic::amdgcn_image_sample_l_2d }, // 248
  { Intrinsic::amdgcn_image_sample_l_2darray }, // 249
  { Intrinsic::amdgcn_image_sample_l_3d }, // 250
  { Intrinsic::amdgcn_image_sample_l_cube }, // 251
  { Intrinsic::amdgcn_image_sample_l_o_1d }, // 252
  { Intrinsic::amdgcn_image_sample_l_o_1darray }, // 253
  { Intrinsic::amdgcn_image_sample_l_o_2d }, // 254
  { Intrinsic::amdgcn_image_sample_l_o_2darray }, // 255
  { Intrinsic::amdgcn_image_sample_l_o_3d }, // 256
  { Intrinsic::amdgcn_image_sample_l_o_cube }, // 257
  { Intrinsic::amdgcn_image_sample_lz_1d }, // 258
  { Intrinsic::amdgcn_image_sample_lz_1darray }, // 259
  { Intrinsic::amdgcn_image_sample_lz_2d }, // 260
  { Intrinsic::amdgcn_image_sample_lz_2darray }, // 261
  { Intrinsic::amdgcn_image_sample_lz_3d }, // 262
  { Intrinsic::amdgcn_image_sample_lz_cube }, // 263
  { Intrinsic::amdgcn_image_sample_lz_o_1d }, // 264
  { Intrinsic::amdgcn_image_sample_lz_o_1darray }, // 265
  { Intrinsic::amdgcn_image_sample_lz_o_2d }, // 266
  { Intrinsic::amdgcn_image_sample_lz_o_2darray }, // 267
  { Intrinsic::amdgcn_image_sample_lz_o_3d }, // 268
  { Intrinsic::amdgcn_image_sample_lz_o_cube }, // 269
  { Intrinsic::amdgcn_image_sample_o_1d }, // 270
  { Intrinsic::amdgcn_image_sample_o_1darray }, // 271
  { Intrinsic::amdgcn_image_sample_o_2d }, // 272
  { Intrinsic::amdgcn_image_sample_o_2darray }, // 273
  { Intrinsic::amdgcn_image_sample_o_3d }, // 274
  { Intrinsic::amdgcn_image_sample_o_cube }, // 275
 };

const AMDGPUImageDMaskIntrinsic *getAMDGPUImageDMaskIntrinsic(unsigned Intr) {
  if ((Intr < Intrinsic::amdgcn_image_getlod_1d) ||
      (Intr > Intrinsic::amdgcn_image_sample_o_cube))
    return nullptr;

  struct KeyType {
    unsigned Intr;
  };
  KeyType Key = { Intr };
  auto Table = makeArrayRef(AMDGPUImageDMaskIntrinsicTable);
  auto Idx = std::lower_bound(Table.begin(), Table.end(), Key,
    [](const AMDGPUImageDMaskIntrinsic &LHS, const KeyType &RHS) {
      if (LHS.Intr < RHS.Intr)
        return true;
      if (LHS.Intr > RHS.Intr)
        return false;
      return false;
    });

  if (Idx == Table.end() ||
      Key.Intr != Idx->Intr)
    return nullptr;
  return &*Idx;
}
#endif

#undef GET_AMDGPUImageDMaskIntrinsicTable_DECL
#undef GET_AMDGPUImageDMaskIntrinsicTable_IMPL
