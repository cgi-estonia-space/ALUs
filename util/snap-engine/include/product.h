#pragma once

#include "dataset.hpp"
#include "geocoding.cuh"

namespace alus {
namespace snapengine {
class Product {
   public:
    alus::snapengine::geocoding::Geocoding *geocoding_;
    alus::Dataset dataset_;
    const char* FILE_FORMAT_;
   private:
};
}
}