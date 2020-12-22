#pragma once

#include <memory>
#include <string>
#include <string_view>

namespace alus {
namespace snapengine {
class MetadataElement;
class Product;
class IMetaDataReader {
protected:
    std::shared_ptr<Product> product_;
    std::string file_name_;

public:
    IMetaDataReader() = default;
    explicit IMetaDataReader(const std::shared_ptr<Product>& product) : product_(product){};
    explicit IMetaDataReader(const std::string_view file_name) : file_name_(file_name){};
    /**
     * Read from file using implementation
     *
     * @param name of root element to be read
     * @return
     */
    [[nodiscard]] virtual std::shared_ptr<MetadataElement> Read(std::string_view name) = 0;
    // in case of default constructor user needs to provide source product
    virtual void SetProduct(const std::shared_ptr<Product>& product) = 0;

    virtual ~IMetaDataReader() = default;
};
}  // namespace snapengine
}  // namespace alus