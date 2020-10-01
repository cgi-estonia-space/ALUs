#include <iostream>

#include "alg_bond.h"
#include "terrain_correction.h"
#include "terrain_correction_metadata.h"

namespace alus::terraincorrection {
class TerrainCorrectionExecuter : public AlgBond {
   public:
    TerrainCorrectionExecuter() { std::cout << __FUNCTION__ << std::endl; };

    int Execute() override {
        std::cout << "Executing TC exec test" << std::endl;

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

    void SetTileSize(size_t width, size_t height) override {
        tile_width_ = width;
        tile_height_ = height;
    }

    void SetOutputFilename(const std::string& output_name) override { output_file_name_ = output_name; }

    ~TerrainCorrectionExecuter() override { std::cout << __FUNCTION__ << std::endl; }

   private:
    std::string input_dataset_name_{};
    std::string metadata_folder_path_{};
    std::string metadata_dim_file_{};
    std::string output_file_name_{};
    size_t tile_width_{};
    size_t tile_height_{};
};
}  // namespace alus::terraincorrection

extern "C" {
alus::AlgBond* CreateAlgorithm() { return new alus::terraincorrection::TerrainCorrectionExecuter(); }

void DeleteAlgorithm(alus::AlgBond* instance) { delete (alus::terraincorrection::TerrainCorrectionExecuter*)instance; }
}
