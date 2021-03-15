#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace alus {
namespace snapengine {
class Assert {
public:
    template <typename T>
    static void NotNull(T object, std::string_view message) {
        if (object == nullptr) {
            throw std::invalid_argument(std::string(message) + " argument is nullptr");
        }
    }

    /**
     * Asserts that an argument is legal. If the given boolean is
     * not <code>true</code>, an <code>IllegalArgumentException</code>
     * is thrown.
     * The given message is included in that exception, to aid debugging.
     *
     * @param expression the outcode of the check
     * @param message    the message to include in the exception
     * @return <code>true</code> if the check passes (does not return
     *         if the check fails)
     * @throws IllegalArgumentException if the legality test failed
     */
    static bool Argument(bool expression, std::string_view message) {
        if (!expression) {
            throw std::invalid_argument(std::string(message).c_str());
        }
        return expression;
    }

    /**
     * Asserts that the given object is not <code>null</code>. If this
     * is not the case, some kind of unchecked exception is thrown.
     *
     * @param object the value to test
     * @throws NullPointerException if the object is <code>null</code>
     */
    template <typename T>
    static void NotNull(T object) {
        NotNull(object, "Assert.notNull(null) called");  //$NON-NLS-1$
    }
};
}  // namespace snapengine
}  // namespace alus
