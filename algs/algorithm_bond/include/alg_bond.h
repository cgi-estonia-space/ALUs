#pragma once

#include <cstddef>

#include "raster_properties.hpp"

namespace alus {
/**
 * Interface for algorithms that could be managed by arbitrary entity above.
 *
 * Algorithms are still able to design their own approach of handling computations.
 * This interface ensures that one can supply inputs and outputs also handle tile sizes etc.
 */
class AlgBond {
   public:
    AlgBond() = default;

    virtual void SetInputs(const std::string& input_dataset, const std::string& metadata_path) = 0;
    virtual void SetTileSize(size_t width, size_t height) = 0;
    virtual void SetOutputFilename(const std::string& output_name) = 0;
    virtual int Execute() = 0;
    [[nodiscard]] virtual RasterDimension CalculateInputTileFrom(RasterDimension output) const = 0;

    virtual ~AlgBond() = default;
};
}  // namespace alus

/**
 * Function signature for CreateAlgorithm()
 */
typedef alus::AlgBond* (*AlgorithmBondEntry)();

/**
 * Function signature for DeleteAlgorithm()
 */
typedef void (*AlgorithmBondEnd)(alus::AlgBond* instance);

extern "C" {
/**
 * Shared library symbol to create an algorithm entity.
 *
 * Symbol is extern "C"-d because C++ symbols are mangled e.g. instead of clean "CreateAlgorithm"
 * it could be "_Z9CreateAlgorithmv" which usage is more error prone. Each algorithm will provide its
 * implementation in respective ".so".
 *
 * @return An algorithm instance
 */
alus::AlgBond* CreateAlgorithm();

/**
 * Shared library symbol to delete an algorithm entity.
 *
 * Symbol is extern "C"-d because C++ symbols are mangled e.g. instead of clean "DeleteAlgorithm"
 * it could be "_Z11CreateAlgorithmiK" which usage is more error prone. Each algorithm will provide its
 * implementation in respective ".so".
 */
void DeleteAlgorithm(alus::AlgBond* instance);
}