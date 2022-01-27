/**
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see http://www.gnu.org/licenses/
 */
#pragma once

#include "alg_bond.h"

#include <string_view>

namespace alus {

/**
 * Takes care of the lifetime of an algorithm instance. Loading shared library from which an instance of AlgBond
 * can be constructed from. Also takes care that instance is deleted together with unloading shared library.
 */
class AlgorithmLoadGuard final {
public:
    using SharedLibHandle = void*;
    using SharedLibSymbol = void*;

    AlgorithmLoadGuard() = delete;

    /**
     * Load respective shared library and create an algorithm instance.
     *
     * @param lib_name Library raw name, without prefix (lib) and postfix (.so)
     * @throws AlgorithmCreationException
     */
    explicit AlgorithmLoadGuard(std::string_view lib_name);

    AlgorithmLoadGuard(const AlgorithmLoadGuard&) = delete;             // Copy ctor.
    AlgorithmLoadGuard& operator=(const AlgorithmLoadGuard&) = delete;  // Copy assignment.

    AlgorithmLoadGuard(AlgorithmLoadGuard&& other) noexcept {  // Move ctor
        this->lib_handle_ = other.lib_handle_;
        other.lib_handle_ = nullptr;
        this->instance_handle_ = other.instance_handle_;
        other.instance_handle_ = nullptr;
        this->instance_dtor_ = other.instance_dtor_;
        other.instance_dtor_ = nullptr;
    }

    AlgorithmLoadGuard& operator=(AlgorithmLoadGuard&& other) noexcept {  // Move assignment
        this->lib_handle_ = other.lib_handle_;
        other.lib_handle_ = nullptr;
        this->instance_handle_ = other.instance_handle_;
        other.instance_handle_ = nullptr;
        this->instance_dtor_ = other.instance_dtor_;
        other.instance_dtor_ = nullptr;
        return *this;
    }

    AlgBond* GetInstanceHandle() const { return instance_handle_; }

    ~AlgorithmLoadGuard();

private:
    /**
     * Load a shared library
     *
     * @param lib_name Library raw name, without prefix (lib) and postfix (.so)
     * @throws AlgorithmCreationException
     * @return A shared library handle
     */
    static SharedLibHandle LoadSharedLibrary(std::string_view lib_name);

    /**
     * Load an algorithm constructor call.
     *
     * @param lib_handle Previously loaded shared library handle.
     * @throws AlgorithmCreationException
     * @return A callback to construct algorithm
     */
    static AlgorithmBondEntry LoadAlgorithmConstructorFrom(SharedLibHandle lib_handle);

    /**
     * Load an algorithm destructor call.
     *
     * @param lib_handle Previously loaded shared library handle.
     * @throws AlgorithmCreationException
     * @return A callback to destruct algorithm
     */
    static AlgorithmBondEnd LoadAlgorithmDestructorFrom(SharedLibHandle lib_handle);

    /**
     * Generic symbol address loading.
     *
     * @param lib_handle Previously loaded shared library handle.
     * @param symbol Symbol name to load from shared library via @lib_handle
     * @throws AlgorithmCreationException
     * @return Symbol address/location
     */
    static SharedLibSymbol LoadSharedLibSymbol(SharedLibHandle lib_handle, std::string_view symbol);

    SharedLibHandle lib_handle_{nullptr};
    AlgBond* instance_handle_{nullptr};
    AlgorithmBondEnd instance_dtor_{nullptr};
};

class AlgorithmCreationException final : public std::runtime_error {
public:
    explicit AlgorithmCreationException(std::string_view err_msg) : std::runtime_error(err_msg.data()) {}
};

}  // namespace alus