#include "product_node.h"

#include <boost/algorithm/string/trim.hpp>

#include "guardian.h"

namespace alus {
namespace snapengine {

void ProductNode::SetDescription(std::string_view description) { this->description_ = description; }
ProductNode::ProductNode(std::string_view name) : ProductNode(name, "") {}

ProductNode::ProductNode(std::string_view name, std::string_view description) {
    std::string name_str{std::string(name)};
    boost::algorithm::trim(name_str);
    Guardian::AssertNotNullOrEmpty("name", name_str);
    name_ = name;
    description_ = description;
}
void ProductNode::SetOwner(const std::shared_ptr<ProductNode>& owner) {
    if (owner != owner_) {
        owner_ = owner;
    }
}

}  // namespace snapengine
}  // namespace alus