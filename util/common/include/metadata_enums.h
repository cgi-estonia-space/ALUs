#pragma once

namespace alus {
namespace metadata {
enum class ProductType { SLC };

enum class AcquisitionMode { IW };

enum class AntennaDirection { RIGHT, LEFT };

enum class Swath { IW1 };

enum class Pass { ASCENDING, DESCENDING };

enum class SampleType { COMPLEX };

enum class Polarisation { VH, VV };

enum class Algorithm { RANGE_DOPPLER };
}  // namespace metadata
}  // namespace alus