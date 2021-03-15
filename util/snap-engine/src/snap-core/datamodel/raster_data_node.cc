#include "raster_data_node.h"

#include <cmath>
#include <stdexcept>

#include "custom/dimension.h"
#include "product.h"
#include "product_node_group.h"

namespace alus::snapengine {

int RasterDataNode::READ_BUFFER_MAX_SIZE = 8 * 1024 * 1024;  // 8 MB

RasterDataNode::RasterDataNode(std::string_view name, int data_type, long num_elems)
    : DataNode(name, data_type, num_elems) {
    if (data_type != ProductData::TYPE_INT8 && data_type != ProductData::TYPE_INT16 &&
        data_type != ProductData::TYPE_INT32 && data_type != ProductData::TYPE_UINT8 &&
        data_type != ProductData::TYPE_UINT16 && data_type != ProductData::TYPE_UINT32 &&
        data_type != ProductData::TYPE_FLOAT32 && data_type != ProductData::TYPE_FLOAT64) {
        throw std::invalid_argument("data_type is invalid");
    }

    scaling_factor_ = 1.0;
    scaling_offset_ = 0.0;
    log10_scaled_ = false;
    scaling_applied_ = false;

    no_data_ = nullptr;
    no_data_value_used_ = false;
    geophysical_no_data_value_ = 0.0;
    valid_pixel_expression_ = "";

    // todo:looks like these are for rendering, should not be important atm.
    //        imageToModelTransform = null;
    //        modelToSceneTransform = MathTransform2D.IDENTITY;
    //        sceneToModelTransform = MathTransform2D.IDENTITY;

//    todo: add support if needed
    //    overlay_masks_ = std::make_shared<ProductNodeGroup<>>(this, "overlayMasks", false);
}
bool RasterDataNode::IsFloatingPointType() { return scaling_applied_ || DataNode::IsFloatingPointType(); }
int RasterDataNode::GetGeophysicalDataType() {
    return 0;
    //    todo: implement if needed
    //    ImageManager->GetProductDataType(
    //        ReinterpretDescriptor->GetTargetDataType(std::shared_ptr<ImageManager>->GetDataBufferType(GetDataType()),
    //                                                 GetScalingFactor(),
    //                                                 GetScalingOffset(),
    //                                                 GetScalingType(),
    //                                                 GetInterpretationType()));
}
void RasterDataNode::SetRasterData(std::shared_ptr<ProductData> raster_data) {
    std::shared_ptr<ProductData> old_data = GetData();
    if (old_data != raster_data) {
        if (raster_data != nullptr) {
            if (raster_data->GetType() != GetDataType()) {
                throw std::invalid_argument("rasterData.getType() != getDataType()");
            }
            if (raster_data->GetNumElems() != GetRasterWidth() * GetRasterHeight()) {
                throw std::invalid_argument("rasterData.getNumElems() != getRasterWidth() * getRasterHeight()");
            }
        }
        SetData(raster_data);
    }
}
void RasterDataNode::SetScalingFactor(double scaling_factor) {
    if (scaling_factor_ != scaling_factor) {
        scaling_factor_ = scaling_factor;
        SetScalingApplied();
        //        ResetGeophysicalImage();
        //            fireProductNodeChanged(PROPERTY_NAME_SCALING_FACTOR);
        SetGeophysicalNoDataValue();
        ResetValidMask();
        SetModified(true);
    }
}
double RasterDataNode::Scale(double v) {
    v = v * scaling_factor_ + scaling_offset_;
    if (log10_scaled_) {
        v = pow(10.0, v);
    }
    return v;
}
void RasterDataNode::SetScalingOffset(double scaling_offset) {
    if (scaling_offset_ != scaling_offset) {
        scaling_offset_ = scaling_offset;
        SetScalingApplied();
        //        ResetGeophysicalImage();
        SetGeophysicalNoDataValue();
        ResetValidMask();
        SetModified(true);
    }
}

void RasterDataNode::SetLog10Scaled(bool log10_scaled) {
    if (log10_scaled_ != log10_scaled) {
        log10_scaled_ = log10_scaled;
        SetScalingApplied();
        //        ResetGeophysicalImage();
        SetGeophysicalNoDataValue();
        ResetValidMask();
        SetModified(true);
    }
}

void RasterDataNode::SetNoDataValueUsed(bool no_data_value_used) {
    if (no_data_value_used_ != no_data_value_used) {
        no_data_value_used_ = no_data_value_used;
        ResetValidMask();
        SetModified(true);
    }
}

void RasterDataNode::SetNoDataValue(double no_data_value) {
    if (no_data_ == nullptr || GetNoDataValue() != no_data_value) {
        if (no_data_ == nullptr) {
            no_data_ = CreateCompatibleProductData(1);
        }
        no_data_->SetElemDouble(no_data_value);
        SetGeophysicalNoDataValue();
        if (IsNoDataValueUsed()) {
            ResetValidMask();
        }
        SetModified(true);
    }
}
void RasterDataNode::SetValidPixelExpression(std::string_view valid_pixel_expression) {
    if (valid_pixel_expression_ != valid_pixel_expression) {
        valid_pixel_expression_ = valid_pixel_expression;
        ResetValidMask();
        SetModified(true);
    }
}
double RasterDataNode::ScaleInverse(double v) {
    if (log10_scaled_) {
        v = log10(v);
    }
    return (v - scaling_offset_) / scaling_factor_;
}
std::shared_ptr<IGeoCoding> RasterDataNode::GetGeoCoding() {
    if (geo_coding_ == nullptr) {
        std::shared_ptr<Product> product = GetProduct();
        if (product) {
            return product->GetSceneGeoCoding();
        }
    }
    return geo_coding_;
}
void RasterDataNode::ReadRasterData(int offset_x, int offset_y, int width, int height,
                                    std::shared_ptr<ProductData> raster_data) {
    ReadRasterData(offset_x, offset_y, width, height, raster_data, nullptr);
}
void RasterDataNode::ReadRasterDataFully() { ReadRasterDataFully(nullptr); }

void RasterDataNode::SetGeoCoding(const std::shared_ptr<IGeoCoding>& geo_coding) {
    if (geo_coding != geo_coding_) {
        geo_coding_ = geo_coding;
        // If our product has no geo-coding yet, it is set to the current one, if any
        if (geo_coding_) {
            std::shared_ptr<Product> product = GetProduct();
            if (product && product->GetSceneGeoCoding() == nullptr &&
                product->GetSceneRasterSize() == GetRasterSize()) {
                product->SetSceneGeoCoding(geo_coding_);
            }
        }
    }
}
std::shared_ptr<custom::Dimension> RasterDataNode::GetRasterSize() {
    return std::make_shared<custom::Dimension>(GetRasterWidth(), GetRasterHeight());
}
void RasterDataNode::SetModified(bool modified) {
    bool old_state = IsModified();
    if (old_state != modified) {
//todo: if overlay_masks get supported, then restore this code
//        if (!modified && overlay_masks_ != nullptr) {
//            overlay_masks_->SetModified(false);
//        }
        ProductNode::SetModified(modified);
    }
}

}  // namespace alus::snapengine