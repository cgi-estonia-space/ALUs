#include "dummy_product_reader.h"

namespace alus::snapengine {
DummyProductReader::DummyProductReader(const std::shared_ptr<IProductReaderPlugIn>& plug_in)
    : AbstractProductReader(plug_in) {}
}  // namespace alus::snapengine