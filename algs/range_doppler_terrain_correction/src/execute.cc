#include <iostream>
#include <string_view>

#include "alg_bond.h"
#include "terrain_correction.h"
#include "terrain_correction_metadata.h"

namespace {
constexpr std::string_view kParameterIdAvgSceneHeight{"avg_scene_height"};
}

namespace alus::terraincorrection {
class TerrainCorrectionExecuter : public AlgBond {
   public:
    TerrainCorrectionExecuter() { std::cout << __FUNCTION__ << std::endl; };

    int Execute() override {
        PrintProcessingParameters();

        Metadata metadata(metadata_dim_file_,
                          metadata_folder_path_ + "/tie_point_grids/latitude.img",
                          metadata_folder_path_ + "/tie_point_grids/longitude.img");
        alus::Dataset input(this->input_dataset_name_);

        TerrainCorrection tc(
            std::move(input), metadata.GetMetadata(), metadata.GetLatTiePoints(), metadata.GetLonTiePoints());
        tc.ExecuteTerrainCorrection(output_file_name_, tile_width_, tile_height_);

        return 0;
    }

    [[nodiscard]] RasterDimension CalculateInputTileFrom(RasterDimension output) const override { return output; }

    void SetInputs(const std::string& input_dataset, const std::string& metadata_path) override {
        input_dataset_name_ = input_dataset;
        metadata_folder_path_ = metadata_path;
        metadata_dim_file_ = metadata_folder_path_.substr(0, metadata_folder_path_.length() - 5) + ".dim";
    }

    void SetParameters(const app::AlgorithmParameters::Table& param_values) override {
        (void) param_values;
    }

    void SetTileSize(size_t width, size_t height) override {
        tile_width_ = width;
        tile_height_ = height;
    }

    void SetOutputFilename(const std::string& output_name) override { output_file_name_ = output_name; }

    [[nodiscard]] std::string GetArgumentsHelp() const override {
        std::stringstream help_stream;
        help_stream << "Range Doppler Terrain Correction configurable parameters:" << std::endl
                    << kParameterIdAvgSceneHeight
                    << " - average scene height to be used instead SRTM values (default:" << avg_scene_height_ << ")"
                    << std::endl;
        return help_stream.str();
    }

    ~TerrainCorrectionExecuter() override { std::cout << __FUNCTION__ << std::endl; }

   private:

    void PrintProcessingParameters() const override {
        std::cout << "Range Doppler Terrain Correction parameters:" << std::endl
        << kParameterIdAvgSceneHeight << " " << avg_scene_height_ << std::endl;
    }

    std::string input_dataset_name_{};
    std::string metadata_folder_path_{};
    std::string metadata_dim_file_{};
    std::string output_file_name_{};
    size_t tile_width_{};
    size_t tile_height_{};
    uint32_t avg_scene_height_{0};
};
}  // namespace alus::terraincorrection

extern "C" {
alus::AlgBond* CreateAlgorithm() { return new alus::terraincorrection::TerrainCorrectionExecuter(); }

void DeleteAlgorithm(alus::AlgBond* instance) { delete (alus::terraincorrection::TerrainCorrectionExecuter*)instance; }
}
