#include <iostream>

#include "alg_bond.h"

namespace alus {
class TerrainCorrectionExecuter : public AlgBond {
   public:
    TerrainCorrectionExecuter() { std::cout << __FUNCTION__ << std::endl; };

    int Execute() override {
        std::cout << "Executing TC exec test" << std::endl;
        return 0;
    }
    [[nodiscard]] RasterDimension CalculateInputTileFrom(RasterDimension output) const override { return output; }
    void SetInputs(const std::string& input_dataset, const std::string& metadata_path) override {
        (void)input_dataset;  // Will be used once TC implementation is done. Currently silencing compiler warnings.
        (void)metadata_path;  // Will be used once TC implementation is done. Currently silencing compiler warnings.
    }

    void SetTileSize(size_t width, size_t height) override {
        (void)width;   // Will be used once TC implementation is done. Currently silencing compiler warnings.
        (void)height;  // Will be used once TC implementation is done. Currently silencing compiler warnings.
    }

    void SetOutputFilename(const std::string& output_name) override {
        (void)output_name;  // Will be used once TC implementation is done. Currently silencing compiler warnings.
    }

    ~TerrainCorrectionExecuter() override { std::cout << __FUNCTION__ << std::endl; }
};
}  // namespace alus

extern "C" {
alus::AlgBond* CreateAlgorithm() { return new alus::TerrainCorrectionExecuter(); }

void DeleteAlgorithm(alus::AlgBond* instance) { delete (alus::TerrainCorrectionExecuter*)instance; }
}
