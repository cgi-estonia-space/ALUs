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
#include "alg_load.h"

#include <dlfcn.h>
#include <cassert>
#include <iostream>
#include <string>
#include <string_view>

namespace alus {

AlgorithmLoadGuard::AlgorithmLoadGuard(std::string_view lib_name) {
    lib_handle_ = LoadSharedLibrary(lib_name);
    assert(lib_handle_ != nullptr);
    auto ctor = LoadAlgorithmConstructorFrom(lib_handle_);
    assert(ctor != nullptr);
    instance_dtor_ = LoadAlgorithmDestructorFrom(lib_handle_);
    assert(instance_dtor_ != nullptr);
    instance_handle_ = ctor();
}

AlgorithmLoadGuard::SharedLibSymbol AlgorithmLoadGuard::LoadSharedLibrary(std::string_view lib_name) {
    const auto file_name = "lib" + std::string(lib_name) + ".so";
    void* alg_lib = dlopen(file_name.c_str(), RTLD_LAZY);
    if (alg_lib == nullptr) {
        throw AlgorithmCreationException("No library '" + std::string(lib_name) + "' found - " + dlerror());
    }

    return alg_lib;
}

AlgorithmLoadGuard::SharedLibSymbol AlgorithmLoadGuard::LoadSharedLibSymbol(
    AlgorithmLoadGuard::SharedLibHandle lib_handle, std::string_view symbol) {
    void* symbol_address = dlsym(lib_handle, symbol.data());
    if (symbol_address == nullptr) {
        throw AlgorithmCreationException("No symbol '" + std::string(symbol) + "' found - " + std::string(dlerror()));
    }

    return symbol_address;
}

AlgorithmBondEntry AlgorithmLoadGuard::LoadAlgorithmConstructorFrom(AlgorithmLoadGuard::SharedLibHandle lib_handle) {
    return (AlgorithmBondEntry)LoadSharedLibSymbol(lib_handle, "CreateAlgorithm");
}

AlgorithmBondEnd AlgorithmLoadGuard::LoadAlgorithmDestructorFrom(AlgorithmLoadGuard::SharedLibHandle lib_handle) {
    return (AlgorithmBondEnd)LoadSharedLibSymbol(lib_handle, "DeleteAlgorithm");
}

AlgorithmLoadGuard::~AlgorithmLoadGuard() {
    if (instance_handle_ != nullptr && instance_dtor_ != nullptr) {
        instance_dtor_(instance_handle_);
    }

    if (lib_handle_ != nullptr) {
        const auto result = dlclose(lib_handle_);
        if (result != 0) {
            std::cerr << "Failure to unload shared library - " << dlerror() << std::endl;
        }
    }
}

}  // namespace alus