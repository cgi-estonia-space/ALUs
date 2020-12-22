#include "snap-engine-utilities/orbit_vector.h"

namespace alus {
namespace snapengine {

int OrbitVector::Compare(const std::shared_ptr<OrbitVector>& osv1, const std::shared_ptr<OrbitVector>& osv2) {
    //    if (osv1->utc_mjd_ < osv2->utc_mjd_) {
    //        return -1;
    //    } else if (osv1->utc_mjd_ > osv2->utc_mjd_) {
    //        return 1;
    //    } else {
    //        return 0;
    //    }
    return static_cast<int>(osv1->utc_mjd_ < osv2->utc_mjd_ && osv2->utc_mjd_ >= osv1->utc_mjd_);
}

}  // namespace snapengine
}  // namespace alus
