#pragma once

#include <memory>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include "s1tbx-io/sentinel1/sentinel1_product_reader_plug_in.h"

namespace alus::snapengine {

// pre-declare
class Product;
class IProductReader;
}  // namespace alus::snapengine

namespace alus::s1tbx {

// pre-declare
class ValidationOptions;

/**
 * Paths to common SAR input data
 */
class ReaderTest : public std::enable_shared_from_this<ReaderTest>  {
protected:
    const std::shared_ptr<Sentinel1ProductReaderPlugIn> reader_plug_in_;
    const std::shared_ptr<snapengine::IProductReader> reader_;
    bool verify_time_ = true;
    bool verify_geocoding_ = true;

    std::shared_ptr<snapengine::Product> TestReader(const boost::filesystem::path& input_path,
                                                    const std::shared_ptr<Sentinel1ProductReaderPlugIn>& reader_plug_in);

    void ValidateMetadata(const std::shared_ptr<snapengine::Product>& product,
                          const std::shared_ptr<ValidationOptions>& options);

    // todo: check what this is
    //    static {
    //        TestUtils.initTestEnvironment();
    //    }

public:

    explicit ReaderTest(const std::shared_ptr<Sentinel1ProductReaderPlugIn>& reader_plug_in);

    std::shared_ptr<snapengine::Product> TestReader(const boost::filesystem::path& input_path);
    void ValidateProduct(const std::shared_ptr<snapengine::Product>& product);
    void ValidateMetadata(const std::shared_ptr<snapengine::Product>& product);
    void ValidateBands(const std::shared_ptr<snapengine::Product>& trg_product, std::vector<std::string> band_names);
};
}  // namespace alus::s1tbx
