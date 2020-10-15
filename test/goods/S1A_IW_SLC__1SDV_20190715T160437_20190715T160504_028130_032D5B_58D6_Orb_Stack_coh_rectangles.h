#pragma once

#include <vector>

#include "shapes.h"

namespace alus::goods {
std::vector<alus::Rectangle> const SOURCE_RECTANGLES{
    {9240, 0, 420, 416},    {9660, 0, 420, 416},    {10080, 0, 420, 416},   {10500, 0, 420, 416},
    {10920, 0, 420, 416},   {11340, 0, 420, 416},   {11760, 0, 420, 416},   {12600, 0, 420, 416},
    {13020, 0, 420, 416},   {5040, 416, 420, 416},  {5460, 416, 420, 416},  {5880, 416, 420, 416},
    {6300, 416, 420, 416},  {6720, 416, 420, 416},  {7140, 416, 420, 416},  {7560, 416, 420, 416},
    {7980, 416, 420, 416},  {8400, 416, 420, 416},  {8820, 416, 420, 416},  {9240, 416, 420, 416},
    {9660, 416, 420, 416},  {10080, 416, 420, 416}, {10500, 416, 420, 416}, {10920, 416, 420, 416},
    {11340, 416, 420, 416}, {11760, 416, 420, 416}, {1680, 832, 420, 416},  {2100, 832, 420, 416},
    {2520, 832, 420, 416},  {2940, 832, 420, 416},  {3360, 832, 420, 416},  {3780, 832, 420, 416},
    {4200, 832, 420, 416},  {4620, 832, 420, 416},  {5040, 832, 420, 416},  {5460, 832, 420, 416},
    {5880, 832, 420, 416},  {6300, 832, 420, 416},  {6720, 832, 420, 416},  {7140, 832, 420, 416},
    {7560, 832, 420, 416},  {7980, 832, 420, 416},  {8400, 832, 420, 416},  {8820, 832, 420, 416},
    {9240, 832, 420, 416},  {9660, 832, 420, 416},  {10080, 832, 420, 416}, {10500, 832, 420, 416},
    {10920, 832, 420, 416}, {11340, 832, 420, 416}, {420, 1248, 420, 416},  {840, 1248, 420, 416},
    {1260, 1248, 420, 416}, {1680, 1248, 420, 416}, {2100, 1248, 420, 416}, {2520, 1248, 420, 416},
    {2940, 1248, 420, 416}, {3360, 1248, 420, 416}, {3780, 1248, 420, 416}, {4200, 1248, 420, 416},
    {4620, 1248, 420, 416}, {5040, 1248, 420, 416}, {5460, 1248, 420, 416}, {5880, 1248, 420, 416},
    {6300, 1248, 420, 416}, {6720, 1248, 420, 416}, {7140, 1248, 420, 416}, {7560, 1248, 420, 416},
    {7980, 1248, 420, 416}, {8400, 1248, 420, 416}, {8820, 1248, 420, 416}, {9240, 1248, 420, 416},
    {9660, 1248, 420, 416}, {0, 1664, 420, 416},    {420, 1664, 420, 416},  {840, 1664, 420, 416},
    {1260, 1664, 420, 416}, {1680, 1664, 420, 416}, {2100, 1664, 420, 416}, {2520, 1664, 420, 416},
    {2940, 1664, 420, 416}, {3360, 1664, 420, 416}, {3780, 1664, 420, 416}, {4200, 1664, 420, 416},
    {4620, 1664, 420, 416}, {5040, 1664, 420, 416}, {5460, 1664, 420, 416}, {5880, 1664, 420, 416},
    {6300, 1664, 420, 416}, {6720, 1664, 420, 416}, {7140, 1664, 420, 416}, {7560, 1664, 420, 416},
    {7980, 1664, 420, 416}, {8400, 1664, 420, 416}, {8820, 1664, 420, 416}, {0, 2080, 420, 416},
    {420, 2080, 420, 416},  {840, 2080, 420, 416},  {1260, 2080, 420, 416}, {1680, 2080, 420, 416},
    {2100, 2080, 420, 416}, {2520, 2080, 420, 416}, {2940, 2080, 420, 416}, {3360, 2080, 420, 416},
    {3780, 2080, 420, 416}, {4200, 2080, 420, 416}, {4620, 2080, 420, 416}, {5040, 2080, 420, 416},
    {5460, 2080, 420, 416}, {5880, 2080, 420, 416}, {6300, 2080, 420, 416}, {6720, 2080, 420, 416},
    {7140, 2080, 420, 416}, {7560, 2080, 420, 416}, {7980, 2080, 420, 416}, {420, 2496, 420, 410},
    {840, 2496, 420, 410},  {1260, 2496, 420, 410}, {1680, 2496, 420, 410}, {2100, 2496, 420, 410},
    {2520, 2496, 420, 410}, {2940, 2496, 420, 410}, {3360, 2496, 420, 410}, {3780, 2496, 420, 410},
    {4200, 2496, 420, 410}};
std::vector<alus::Rectangle> const EXPECTED_RECTANGLES{
    {15928, 1459, 552, 36},  {16478, 1458, 4, 4},     {18015, 1473, 4, 4},     {18017, 1340, 564, 137},
    {19322, 1398, 4, 4},     {19695, 1255, 634, 228}, {20257, 1335, 452, 116}, {22289, 1331, 383, 117},
    {22669, 1187, 609, 300}, {9055, 1493, 4, 4},      {9056, 1448, 698, 49},   {9751, 1403, 705, 49},
    {10454, 1358, 772, 139}, {11155, 1314, 781, 149}, {11862, 1270, 857, 228}, {12575, 1226, 858, 250},
    {13289, 1182, 936, 318}, {14012, 1138, 935, 352}, {14734, 1094, 944, 353}, {15462, 1051, 1017, 443},
    {16195, 1008, 949, 454}, {16929, 964, 810, 352},  {17668, 921, 842, 444},  {18410, 900, 376, 126},
    {19928, 1141, 190, 117}, {20116, 1130, 192, 15},  {2871, 1452, 488, 38},   {3356, 1405, 666, 51},
    {4020, 1359, 742, 139},  {4688, 1313, 742, 151},  {5355, 1268, 814, 229},  {6029, 1222, 819, 252},
    {6711, 1176, 892, 320},  {7399, 1131, 896, 353},  {8083, 1086, 975, 410},  {8778, 1041, 975, 455},
    {9471, 996, 984, 455},   {10175, 951, 982, 455},  {10877, 907, 987, 454},  {11585, 863, 992, 454},
    {12296, 818, 995, 455},  {13013, 774, 1001, 455}, {13731, 730, 1005, 455}, {14452, 687, 1011, 454},
    {15183, 643, 1014, 454}, {15913, 643, 1018, 411}, {16718, 658, 951, 353},  {17455, 625, 957, 342},
    {18338, 593, 677, 331},  {19012, 674, 4, 4},      {945, 1198, 389, 295},   {1198, 1139, 785, 354},
    {1776, 1092, 934, 408},  {2428, 1046, 930, 454},  {3078, 1000, 944, 455},  {3744, 953, 946, 455},
    {4414, 907, 943, 455},   {5086, 861, 945, 455},   {5759, 816, 954, 455},   {6439, 770, 962, 455},
    {7123, 725, 962, 454},   {7808, 679, 972, 455},   {8501, 634, 972, 455},   {9193, 589, 983, 455},
    {9895, 545, 984, 454},   {10599, 500, 988, 454},  {11307, 455, 991, 455},  {12019, 411, 996, 455},
    {12733, 378, 1000, 443}, {13590, 323, 864, 454},  {14173, 301, 1012, 432}, {15042, 358, 873, 332},
    {15913, 642, 4, 4},      {0, 828, 353, 217},      {218, 780, 848, 421},    {858, 734, 920, 443},
    {1504, 687, 925, 455},   {2148, 640, 932, 455},   {2808, 594, 937, 455},   {3475, 547, 941, 455},
    {4147, 501, 941, 455},   {4816, 455, 945, 455},   {5491, 409, 950, 455},   {6167, 364, 957, 455},
    {6850, 318, 960, 455},   {7538, 273, 964, 455},   {8226, 228, 969, 454},   {8918, 182, 979, 455},
    {9614, 138, 986, 454},   {10320, 93, 988, 454},   {11028, 48, 993, 455},   {11740, 4, 994, 454},
    {12454, 4, 931, 410},    {13383, 264, 792, 95},   {14172, 300, 369, 26},   {0, 422, 219, 456},
    {0, 375, 860, 456},      {592, 328, 913, 455},    {1237, 281, 913, 455},   {1882, 234, 927, 456},
    {2536, 188, 941, 455},   {3198, 141, 951, 456},   {3876, 95, 942, 455},    {4545, 49, 947, 455},
    {5219, 3, 950, 455},     {5898, 3, 954, 409},     {6645, 12, 894, 355},    {7330, 1, 897, 320},
    {8088, 23, 832, 253},    {8781, 1, 835, 229},     {9549, 35, 772, 150},    {10250, 1, 780, 140},
    {11027, 47, 714, 49},    {11739, 3, 716, 48},     {12453, 3, 4, 4},        {0, 199, 594, 226},
    {525, 29, 714, 349},     {1039, 5, 845, 326},     {1750, 34, 788, 250},    {2407, 0, 793, 237},
    {3129, 41, 749, 150},    {3810, 6, 737, 138},     {4544, 48, 677, 50},     {5218, 2, 682, 50},
    {5897, 2, 4, 4}};
}  // namespace alus::goods