//#include "mask.h"
//
//namespace alus {
//namespace snapengine {
//
//Mask::Mask(std::string_view name, int width, int height, std::shared_ptr<ImageType> image_type)
//    : Band(name, ProductData::TYPE_UINT8, width, height) {
//    Assert::NotNull(image_type, "imageType");
//    image_type_ = image_type;
//    image_config_listener_ = evt->{
//        if (IsSourceImageSet()) {
//            // Added setSourceImage(null), otherwise
//            // org.esa.snap.core.datamodel.MaskTest.testReassignExpression
//            // cannot work. (nf 2015-07-27)
//            //
//            std::shared_ptr<MultiLevelImage> source_image = GetSourceImage();
//            SetSourceImage(nullptr);
//            // The sourceImage.reset() call is left here
//            // so that old level images are removed from JAI tile cache.
//            source_image->Reset();
//        }
//        //            fireProductNodeChanged(evt.getPropertyName(), evt.getOldValue(), evt.getNewValue());
//    };
//    image_config_ = image_type.CreateImageConfig();
//    //        image_config_->AddPropertyChangeListener(image_config_listener_);
//}
//}  // namespace snapengine
//}  // namespace alus