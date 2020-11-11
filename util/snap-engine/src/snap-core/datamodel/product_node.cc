#include "product_node.h"

#include <boost/algorithm/string/trim.hpp>

#include "guardian.h"
#include "product.h"

namespace alus {
namespace snapengine {

void ProductNode::SetDescription(std::string_view description) { description_ = description; }
void ProductNode::SetDescription(const std::optional<std::string_view>& description) { description_ = description; }

ProductNode::ProductNode(std::string_view name) : ProductNode(name, {}) {
}

ProductNode::ProductNode(std::string_view name, const std::optional<std::string_view>& description) {
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
std::shared_ptr<Product> ProductNode::GetProduct() {
    //            todo:add support for thread safty (java had synchronized)
    // todo: also check if instanceof logic from java has been replaced like needed
    if (product_ == nullptr) {
        std::shared_ptr<ProductNode> owner = shared_from_this();
        do {
//            todo:this might need to check if !check_type.empty()
            auto check_type = std::dynamic_pointer_cast<Product>(owner);
            if (check_type != nullptr) {
                product_ = check_type;
                break;
            }
            owner = owner->GetOwner();
        } while (owner != nullptr);
    }
    return product_;
}

void ProductNode::SetModified(bool modified) {
    bool old_state = modified_;
    if (old_state != modified) {
        modified_ = modified;
        // If this node is modified, the owner is also modified.
        if (modified_ && GetOwner() != nullptr) {
            GetOwner()->SetModified(true);
        }
    }
}

}  // namespace snapengine
}  // namespace alus