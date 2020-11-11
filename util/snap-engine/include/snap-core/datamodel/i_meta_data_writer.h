#pragma once

#include <memory>

namespace alus {
namespace snapengine {
class Product;

/**
 * Temporary abstract class which provides API for implementations of metadata writers from internal root element.
 */
class IMetaDataWriter {
   protected:
    std::shared_ptr<Product> product_;

   public:
    IMetaDataWriter() = default;
    IMetaDataWriter(const std::shared_ptr<Product>& product) : product_(product){};
    // write to file
    virtual void Write() = 0;
    // in case of default constructor user needs to provide target product
    virtual void SetProduct(const std::shared_ptr<Product>& product) = 0;
    virtual ~IMetaDataWriter() = default;
};
}  // namespace snapengine
}  // namespace alus