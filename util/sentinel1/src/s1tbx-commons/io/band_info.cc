#include "s1tbx-commons/io/band_info.h"

#include "snap-engine-utilities/datamodel/unit.h"

namespace alus::s1tbx {

BandInfo::BandInfo(const std::shared_ptr<snapengine::Band>& band, const std::shared_ptr<ImageIOFile>& img_file,
                   const int id, const int offset)
    : image_i_d_(id),
      band_sample_offset_(offset),
      img_(img_file),
      is_imaginary_(band->GetUnit().has_value() && (band->GetUnit().value() == snapengine::Unit::IMAGINARY)) {
    if (is_imaginary_) {
        imaginary_band_ = band;
    } else {
        real_band_ = band;
    }
}
}  // namespace alus::s1tbx
