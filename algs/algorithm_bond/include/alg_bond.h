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

#include <cstddef>
#include <string>
#include <vector>

#include <gdal_priv.h>

#include "algorithm_parameters.h"

namespace alus {

// Forward declaration for non DEM and EGM96 applicable entities - no need to mess with includes.
namespace snapengine {
class Srtm3ElevationModel;
class EarthGravitationalModel96;
}

/**
 * Interface for algorithms that could be managed by arbitrary entity above.
 *
 * Algorithms are still able to design their own approach of handling computations.
 * This interface ensures that one can supply inputs and outputs also handle tile sizes etc.
 */
class AlgBond {
public:
    AlgBond() = default;
    AlgBond(const AlgBond&) = delete;
    AlgBond& operator=(const AlgBond&) = delete;

    virtual ~AlgBond() = default;

    virtual void SetInputFilenames([[maybe_unused]] const std::vector<std::string>& input_datasets,
                                   [[maybe_unused]] const std::vector<std::string>&
                                       metadata_paths) { /* Either one of the SetInputFilenames() or SetInputDataset()
                                                           must be implemented, can be both too.*/
    }
    virtual void SetInputDataset([[maybe_unused]] const std::vector<GDALDataset*>& inputs,
                                 [[maybe_unused]] const std::vector<std::string>&
                                     metadata_paths) { /* Either one of the SetInputFilenames() or SetInputDataset() must
                                                         be implemented, can be both too.*/
    }
    virtual void SetParameters(const app::AlgorithmParameters::Table& param_values) = 0;
    virtual void SetSrtm3Manager(snapengine::Srtm3ElevationModel* /*manager*/) { /* Some of the operators do not require
                                                                                    elevation calulations. */
    }
    virtual void SetEgm96Manager(
        const snapengine::EarthGravitationalModel96* /*manager*/) { /* Some of the operators do not require elevation
                                                                       calulations. */
    }
    virtual void SetTileSize(size_t width, size_t height) = 0;
    virtual void SetOutputFilename(
        [[maybe_unused]] const std::string& output_name) { /* Either one of the SetOutputFilename() or SetOutputDriver()
                                                              must be implemented, can be both too. */
    }
    virtual void SetOutputDriver(
        [[maybe_unused]] GDALDriver* output_driver) { /* Either one of the SetOutputFilename() or SetOutputDriver() must
                                                         be implemented, can be both too. */
    }
    [[nodiscard]] virtual int Execute() = 0;
    [[nodiscard]] virtual GDALDataset* GetProcessedDataset() const { return nullptr; }

    [[nodiscard]] virtual std::string GetArgumentsHelp() const = 0;

private:
    virtual void PrintProcessingParameters() const {}
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
