#include "virtual_band.h"

namespace alus::snapengine {

VirtualBand::VirtualBand(std::string_view name, int data_type, int width, int height, std::string_view expression)
    : Band(name, data_type, width, height) {
    SetSpectralBandIndex(-1);
    SetSynthetic(true);
    expression_ = expression;
}
void VirtualBand::SetExpression(std::string_view expression) {
    if (expression != nullptr && expression_ != expression) {
        expression_ = expression;
//        if (IsSourceImageSet()) {
//            SetSourceImage(nullptr);
//        }
//        these are mostly used for visuals
//        ResetValidMask();
//        SetStx(nullptr);
//        SetImageInfo(nullptr);
        SetModified(true);
        //not supporting/porting events
        //        fireProductNodeChanged(PROPERTY_NAME_EXPRESSION);
        //        fireProductNodeChanged(PROPERTY_NAME_DATA);
        //        fireProductNodeDataChanged();
    }
}

}  // namespace alus::snapengine