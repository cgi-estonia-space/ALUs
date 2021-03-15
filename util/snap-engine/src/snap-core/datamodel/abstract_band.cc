#include "abstract_band.h"

#include <algorithm>
#include <any>
#include <cmath>
#include <stdexcept>
#include <string>

#include "ceres-core/i_progress_monitor.h"
#include "guardian.h"
#include "snap-core/datamodel/product_data.h"

namespace alus {
namespace snapengine {

AbstractBand::AbstractBand(std::string_view name, int data_type, int raster_width, int raster_height)
    : RasterDataNode(name, data_type, static_cast<long>(raster_width) * static_cast<long>(raster_height)) {
    raster_width_ = raster_width;
    raster_height_ = raster_height;
}

void AbstractBand::WritePixels(int x, int y, int w, int h, std::vector<int> pixels,
                               std::shared_ptr<ceres::IProgressMonitor> pm) {
    //    todo: solve guardian issue
    //    Guardian::AssertNotNull("pixels", pixels);
    std::shared_ptr<ProductData> sub_raster_data = CreateCompatibleRasterData(w, h);
    int n = w * h;
    // todo: might need to match types from productdata
    // note: initial version just tries to make a close port of snap java version  optimize later if its slow
    if (!IsScalingApplied() && (sub_raster_data->GetElems()).type() == typeid(std::vector<int>)) {
        auto data = std::any_cast<std::vector<int>>(sub_raster_data->GetElems());
        std::copy(pixels.begin(), pixels.end(), data.begin());
    } else {
        if (IsScalingApplied()) {
            for (int i = 0; i < n; i++) {
                sub_raster_data->SetElemDoubleAt(i, ScaleInverse(pixels.at(i)));
            }
        } else {
            for (int i = 0; i < n; i++) {
                sub_raster_data->SetElemIntAt(i, pixels.at(i));
            }
        }
    }
    WriteRasterData(x, y, w, h, sub_raster_data, pm);
}

void AbstractBand::WritePixels(int x, int y, int w, int h, std::vector<float> pixels,
                               std::shared_ptr<ceres::IProgressMonitor> pm) {
    //    todo: solve guardian issue
    //    Guardian::AssertNotNull("pixels", pixels);
    std::shared_ptr<ProductData> sub_raster_data = CreateCompatibleRasterData(w, h);
    int n = w * h;
    if (!IsScalingApplied() && (sub_raster_data->GetElems()).type() == typeid(std::vector<float>)) {
        auto data = std::any_cast<std::vector<float>>(sub_raster_data->GetElems());
        std::copy(pixels.begin(), pixels.end(), data.begin());
    } else {
        if (IsScalingApplied()) {
            for (int i = 0; i < n; i++) {
                sub_raster_data->SetElemDoubleAt(i, ScaleInverse(pixels.at(i)));
            }
        } else {
            for (int i = 0; i < n; i++) {
                sub_raster_data->SetElemFloatAt(i, pixels.at(i));
            }
        }
    }
    WriteRasterData(x, y, w, h, sub_raster_data, pm);
}
void AbstractBand::WritePixels(int x, int y, int w, int h, std::vector<double> pixels,
                               std::shared_ptr<ceres::IProgressMonitor> pm) {
    //    todo: solve guardian issue
    //    Guardian::AssertNotNull("pixels", pixels);
    std::shared_ptr<ProductData> sub_raster_data = CreateCompatibleRasterData(w, h);
    int n = w * h;
    if (!IsScalingApplied() && (sub_raster_data->GetElems()).type() == typeid(std::vector<double>)) {
        auto data = std::any_cast<std::vector<double>>(sub_raster_data->GetElems());
        std::copy(pixels.begin(), pixels.end(), data.begin());
    } else {
        if (IsScalingApplied()) {
            for (int i = 0; i < n; i++) {
                sub_raster_data->SetElemDoubleAt(i, ScaleInverse(pixels.at(i)));
            }
        } else {
            for (int i = 0; i < n; i++) {
                sub_raster_data->SetElemDoubleAt(i, pixels.at(i));
            }
        }
    }
    WriteRasterData(x, y, w, h, sub_raster_data, pm);
}
int AbstractBand::GetPixelInt(int x, int y) {
    if (IsScalingApplied()) {
        return static_cast<int>(round(Scale(GetRasterData()->GetElemDoubleAt(GetRasterWidth() * y + x))));
    }
    return GetRasterData()->GetElemIntAt(GetRasterWidth() * y + x);
}
float AbstractBand::GetPixelFloat(int x, int y) {
    if (IsScalingApplied()) {
        return static_cast<float>(Scale(GetRasterData()->GetElemDoubleAt(GetRasterWidth() * y + x)));
    }
    return GetRasterData()->GetElemFloatAt(GetRasterWidth() * y + x);
}
double AbstractBand::GetPixelDouble(int x, int y) {
    if (IsScalingApplied()) {
        return Scale(GetRasterData()->GetElemDoubleAt(GetRasterWidth() * y + x));
    }
    return GetRasterData()->GetElemDoubleAt(GetRasterWidth() * y + x);
}
void AbstractBand::SetPixelInt(int x, int y, int pixel_value) {
    if (IsScalingApplied()) {
        GetRasterData()->SetElemDoubleAt(GetRasterWidth() * y + x, ScaleInverse(pixel_value));
    } else {
        GetRasterData()->SetElemIntAt(GetRasterWidth() * y + x, pixel_value);
    }
    SetModified(true);
}
void AbstractBand::SetPixelFloat(int x, int y, float pixel_value) {
    if (IsScalingApplied()) {
        GetRasterData()->SetElemDoubleAt(GetRasterWidth() * y + x, ScaleInverse(pixel_value));
    } else {
        GetRasterData()->SetElemFloatAt(GetRasterWidth() * y + x, pixel_value);
    }
    SetModified(true);
}
void AbstractBand::SetPixelDouble(int x, int y, double pixel_value) {
    if (IsScalingApplied()) {
        GetRasterData()->SetElemDoubleAt(GetRasterWidth() * y + x, ScaleInverse(pixel_value));
    } else {
        GetRasterData()->SetElemDoubleAt(GetRasterWidth() * y + x, pixel_value);
    }
    SetModified(true);
}
void AbstractBand::SetPixels(int x, int y, int w, int h, std::vector<int> pixels) {
    //    Guardian::AssertNotNull("pixels", pixels);
    std::shared_ptr<ProductData> raster_data = GetRasterData();
    int x1 = x;
    int y1 = y;
    int x2 = x1 + w - 1;
    int y2 = y1 + h - 1;
    int pos = 0;
    if (IsScalingApplied()) {
        for (y = y1; y <= y2; y++) {
            int x_offs = y * GetRasterWidth();
            for (x = x1; x <= x2; x++) {
                raster_data->SetElemDoubleAt(x_offs + x, ScaleInverse(pixels.at(pos++)));
            }
        }
    } else {
        for (y = y1; y <= y2; y++) {
            int x_offs = y * GetRasterWidth();
            for (x = x1; x <= x2; x++) {
                raster_data->SetElemIntAt(x_offs + x, pixels.at(pos++));
            }
        }
    }
    SetModified(true);
}
void AbstractBand::SetPixels(int x, int y, int w, int h, std::vector<float> pixels) {
    //    Guardian::AssertNotNull("pixels", pixels);
    std::shared_ptr<ProductData> raster_data = GetRasterData();
    int x1 = x;
    int y1 = y;
    int x2 = x1 + w - 1;
    int y2 = y1 + h - 1;
    int pos = 0;
    if (IsScalingApplied()) {
        for (y = y1; y <= y2; y++) {
            int x_offs = y * GetRasterWidth();
            for (x = x1; x <= x2; x++) {
                raster_data->SetElemDoubleAt(x_offs + x, ScaleInverse(pixels.at(pos++)));
            }
        }
    } else {
        for (y = y1; y <= y2; y++) {
            int x_offs = y * GetRasterWidth();
            for (x = x1; x <= x2; x++) {
                raster_data->SetElemFloatAt(x_offs + x, pixels.at(pos++));
            }
        }
    }
    SetModified(true);
}
void AbstractBand::SetPixels(int x, int y, int w, int h, std::vector<double> pixels) {
    //    Guardian::AssertNotNull("pixels", pixels);
    std::shared_ptr<ProductData> raster_data = GetRasterData();
    int x1 = x;
    int y1 = y;
    int x2 = x1 + w - 1;
    int y2 = y1 + h - 1;
    int pos = 0;
    if (IsScalingApplied()) {
        for (y = y1; y <= y2; y++) {
            int x_offs = y * GetRasterWidth();
            for (x = x1; x <= x2; x++) {
                raster_data->SetElemDoubleAt(x_offs + x, ScaleInverse(pixels.at(pos++)));
            }
        }

    } else {
        for (y = y1; y <= y2; y++) {
            int x_offs = y * GetRasterWidth();
            for (x = x1; x <= x2; x++) {
                raster_data->SetElemDoubleAt(x_offs + x, pixels.at(pos++));
            }
        }
    }
    SetModified(true);
}
std::vector<int> AbstractBand::ReadPixels(int x, int y, int w, int h, std::vector<int> pixels,
                                          std::shared_ptr<ceres::IProgressMonitor> pm) {
    if (HasRasterData()) {
        pixels = GetPixels(x, y, w, h, pixels, pm);
    } else {
        std::shared_ptr<ProductData> raw_data = ReadSubRegionRasterData(x, y, w, h, pm);
        int n = w * h;
        pixels = EnsureMinLengthArray(pixels, n);
        if (!IsScalingApplied() && (raw_data->GetElems()).type() == typeid(std::vector<int>)) {
            auto data = std::any_cast<std::vector<int>>(raw_data->GetElems());
            std::copy(data.begin(), data.end(), pixels.begin());
        } else {
            if (IsScalingApplied()) {
                for (int i = 0; i < n; i++) {
                    pixels.at(i) = static_cast<int>(round(Scale(raw_data->GetElemDoubleAt(i))));
                }
            } else {
                for (int i = 0; i < n; i++) {
                    pixels.at(i) = raw_data->GetElemIntAt(i);
                }
            }
        }
    }
    return pixels;
}
std::vector<float> AbstractBand::ReadPixels(int x, int y, int w, int h, std::vector<float> pixels,
                                            [[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) {
    //        try {
    if (HasRasterData()) {
        //                pm.beginTask("Reading pixels...", 1);
        pixels = GetPixels(x, y, w, h, pixels, nullptr /*, SubProgressMonitor.create(pm, 1)*/);
    } else {
        //                pm.beginTask("Reading pixels...", 2);
        std::shared_ptr<ProductData> raw_data =
            ReadSubRegionRasterData(x, y, w, h, nullptr /*SubProgressMonitor.create(pm, 1)*/);
        int n = w * h;
        pixels = EnsureMinLengthArray(pixels, n);
        if (!IsScalingApplied() && (raw_data->GetElems()).type() == typeid(std::vector<float>)) {
            auto data = std::any_cast<std::vector<float>>(raw_data->GetElems());
            std::copy(data.begin(), data.end(), pixels.begin());
        } else {
            if (IsScalingApplied()) {
                for (int i = 0; i < n; i++) {
                    pixels.at(i) = static_cast<float>(Scale(raw_data->GetElemFloatAt(i)));
                }
            } else {
                for (int i = 0; i < n; i++) {
                    pixels.at(i) = raw_data->GetElemFloatAt(i);
                }
            }
        }
        //                pm.worked(1);
    }
    //        } finally {
    //            pm.done();
    //        }
    return pixels;
}
std::vector<double> AbstractBand::ReadPixels(int x, int y, int w, int h, std::vector<double> pixels,
                                             std::shared_ptr<ceres::IProgressMonitor> pm) {
    if (HasRasterData()) {
        pixels = GetPixels(x, y, w, h, pixels, pm);
    } else {
        std::shared_ptr<ProductData> raw_data = ReadSubRegionRasterData(x, y, w, h, pm);
        int n = w * h;
        pixels = EnsureMinLengthArray(pixels, n);
        if (!IsScalingApplied() && (raw_data->GetElems()).type() == typeid(std::vector<double>)) {
            auto data = std::any_cast<std::vector<double>>(raw_data->GetElems());
            std::copy(data.begin(), data.end(), pixels.begin());
        } else {
            if (IsScalingApplied()) {
                for (int i = 0; i < n; i++) {
                    pixels.at(i) = Scale(raw_data->GetElemDoubleAt(i));
                }
            } else {
                for (int i = 0; i < n; i++) {
                    pixels.at(i) = raw_data->GetElemDoubleAt(i);
                }
            }
        }
    }
    return pixels;
}

std::vector<int> AbstractBand::GetPixels(int x, int y, int w, int h, std::vector<int> pixels,
                                         [[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) {
    pixels = EnsureMinLengthArray(pixels, w * h);
    std::shared_ptr<ProductData> raster_data = GetRasterDataSafe();
    int x1 = x;
    int y1 = y;
    int x2 = x1 + w - 1;
    int y2 = y1 + h - 1;
    int pos = 0;
    //        pm.beginTask("Retrieving pixels...", y2 - y1);
    //        try {
    if (IsScalingApplied()) {
        for (y = y1; y <= y2; y++) {
            int x_offs = y * GetRasterWidth();
            for (x = x1; x <= x2; x++) {
                pixels.at(pos++) = static_cast<int>(round(Scale(raster_data->GetElemDoubleAt(x_offs + x))));
            }
            //                    pm.worked(1);
        }
    } else {
        for (y = y1; y <= y2; y++) {
            int x_offs = y * GetRasterWidth();
            for (x = x1; x <= x2; x++) {
                pixels.at(pos++) = raster_data->GetElemIntAt(x_offs + x);
            }
            //                    pm.worked(1);
        }
    }
    //        } finally {
    //            pm.done();
    //        }
    return pixels;
}
std::vector<float> AbstractBand::GetPixels(int x, int y, int w, int h, std::vector<float> pixels,
                                           [[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) {
    pixels = EnsureMinLengthArray(pixels, w * h);
    std::shared_ptr<ProductData> raster_data = GetRasterDataSafe();
    int x1 = x;
    int y1 = y;
    int x2 = x1 + w - 1;
    int y2 = y1 + h - 1;
    int pos = 0;
    //        pm.beginTask("Retrieving pixels...", y2 - y1);
    //        try {
    if (IsScalingApplied()) {
        for (y = y1; y <= y2; y++) {
            int x_offs = y * GetRasterWidth();
            for (x = x1; x <= x2; x++) {
                pixels.at(pos++) = static_cast<float>(Scale(raster_data->GetElemFloatAt(x_offs + x)));
            }
            //                    pm.worked(1);
        }
    } else {
        for (y = y1; y <= y2; y++) {
            int x_offs = y * GetRasterWidth();
            for (x = x1; x <= x2; x++) {
                pixels.at(pos++) = raster_data->GetElemFloatAt(x_offs + x);
            }
            //                    pm.worked(1);
        }
    }
    //        } finally {
    //            pm.done();
    //        }
    return pixels;
}

std::vector<double> AbstractBand::GetPixels(int x, int y, int w, int h, std::vector<double> pixels,
                                            [[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) {
    pixels = EnsureMinLengthArray(pixels, w * h);
    std::shared_ptr<ProductData> raster_data = GetRasterDataSafe();
    int x1 = x;
    int y1 = y;
    int x2 = x1 + w - 1;
    int y2 = y1 + h - 1;
    int pos = 0;
    //        pm.beginTask("Retrieving pixels...", y2 - y1);
    //        try {
    if (IsScalingApplied()) {
        for (y = y1; y <= y2; y++) {
            //                    if (pm.isCanceled()) {
            //                        break;
            //                    }
            int x_offs = y * GetRasterWidth();
            for (x = x1; x <= x2; x++) {
                pixels.at(pos++) = Scale(raster_data->GetElemDoubleAt(x_offs + x));
            }
            //                    pm.worked(1);
        }
    } else {
        for (y = y1; y <= y2; y++) {
            //                    if (pm.isCanceled()) {
            //                        break;
            //                    }
            int x_offs = y * GetRasterWidth();
            for (x = x1; x <= x2; x++) {
                pixels.at(pos++) = raster_data->GetElemDoubleAt(x_offs + x);
            }
            //                    pm.worked(1);
        }
    }
    //        } finally {
    //            pm.done();
    //        }
    return pixels;
}

std::vector<int> AbstractBand::EnsureMinLengthArray(std::vector<int> array, int length) {
    if (array.empty()) {
        array.resize(length);
        return array;
    }
    if (array.size() < static_cast<size_t>(length)) {
        throw std::invalid_argument("The length of the given array is less than " + std::to_string(length));
    }
    return array;
}

std::vector<float> AbstractBand::EnsureMinLengthArray(std::vector<float> array, int length) {
    if (array.empty()) {
        array.resize(length);
        return array;
    }
    if (array.size() < static_cast<size_t>(length)) {
        throw std::invalid_argument("The length of the given array is less than " + std::to_string(length));
    }
    return array;
}

std::vector<double> AbstractBand::EnsureMinLengthArray(std::vector<double> array, int length) {
    if (array.empty()) {
        array.resize(length);
        return array;
    }
    if (array.size() < static_cast<size_t>(length)) {
        throw std::invalid_argument("The length of the given array is less than " + std::to_string(length));
    }
    return array;
}

std::shared_ptr<ProductData> AbstractBand::GetRasterDataSafe() {
    if (!HasRasterData()) {
        throw std::runtime_error("raster data not loaded");
    }
    return GetRasterData();
}

std::shared_ptr<ProductData> AbstractBand::ReadSubRegionRasterData(int x, int y, int w, int h,
                                                                   std::shared_ptr<ceres::IProgressMonitor> pm) {
    std::shared_ptr<ProductData> sub_raster_data = CreateCompatibleRasterData(w, h);
    ReadRasterData(x, y, w, h, sub_raster_data, pm);
    return sub_raster_data;
}

}  // namespace snapengine
}  // namespace alus
