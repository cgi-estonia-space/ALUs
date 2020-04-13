#include "target_dataset.hpp"

#include "gdal_util.hpp"

namespace slap {

TargetDataset::TargetDataset(Dataset& dsWithProperties, std::string_view filename)
    : dimensions{dsWithProperties.getRasterDimensions()} {
    auto const driver = dsWithProperties.getGDALDataset()->GetDriver();
    CHECK_GDAL_PTR(driver);

    this->gdalDs = driver->Create(filename.data(), dimensions.columnsX, dimensions.rowsY, 1, GDT_Float64, nullptr);
    CHECK_GDAL_PTR(this->gdalDs);
    double gt[GeoTransformConstruct::GDAL_GEOTRANSFORM_PARAMETERS_LENGTH];
    auto inputGt = dsWithProperties.getTransform();
    std::copy_n(inputGt, GeoTransformConstruct::GDAL_GEOTRANSFORM_PARAMETERS_LENGTH, gt);
    CHECK_GDAL_ERROR(this->gdalDs->SetGeoTransform(gt));
    CHECK_GDAL_ERROR(this->gdalDs->SetProjection(dsWithProperties.getGDALDataset()->GetProjectionRef()));
}

void TargetDataset::write(std::vector<double>& from, RasterPoint to, RasterDimension howMuch) {
    int width = this->dimensions.columnsX;
    int height = this->dimensions.rowsY;
    if (howMuch.columnsX != 0 || howMuch.rowsY != 0) {
        width = howMuch.columnsX;
        height = howMuch.rowsY;
    }

    if (static_cast<int>(from.size()) < width * height) {
        throw std::invalid_argument("Writing from input buffer with size " + std::to_string(from.size()) +
                                    " would overflow for dimensions " + std::to_string(width) + "x" +
                                    std::to_string(height));
    }

    CHECK_GDAL_ERROR(this->gdalDs->GetRasterBand(1)->RasterIO(
        GF_Write, to.x, to.y, width, height, from.data(), width, height, GDT_Float64, 0, 0));
}

TargetDataset::~TargetDataset() {
    if (this->gdalDs) {
        GDALClose(this->gdalDs);
        this->gdalDs = nullptr;
    }
}
}  // namespace slap