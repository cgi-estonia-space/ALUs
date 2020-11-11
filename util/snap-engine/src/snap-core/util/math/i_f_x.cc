#include "snap-core/util/math/i_f_x.h"

#include "snap-core/util/math/functions.h"

namespace alus {
namespace snapengine {

const std::reference_wrapper<IFX> IFX::XXXX = *new functions::FX_X4();
const std::reference_wrapper<IFX> IFX::XXX = *new functions::FX_X3();
const std::reference_wrapper<IFX> IFX::XX = *new functions::FX_X2();
const std::reference_wrapper<IFX> IFX::X = *new functions::FX_X();
const std::reference_wrapper<IFX> IFX::ONE = *new functions::FX_1();

IFX::~IFX() {}

}  // namespace snapengine
}  // namespace alus
