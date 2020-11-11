#include "dummy_product_reader_plug_in.h"

namespace alus::snapengine {
std::shared_ptr<IProductReader> DummyProductReaderPlugIn::CreateReaderInstance() {
    return std::make_shared<DummyProductReader>(shared_from_this());
}
}  // namespace alus::snapengine