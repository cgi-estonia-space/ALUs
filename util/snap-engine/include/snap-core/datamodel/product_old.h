#pragma once

#include "dataset.h"
#include "geocoding.h"

namespace alus {
namespace snapengine {
namespace old {

class Product {
   public:
    alus::snapengine::geocoding::Geocoding *geocoding_;
    alus::Dataset dataset_;
    const char* FILE_FORMAT_;
   private:
};
}
}
}