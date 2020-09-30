#include <fstream>
#include <iostream>

#include "alg_bond.h"
#include "terrain_correction.hpp"

namespace alus {
class TerrainCorrectionExecuter : public AlgBond {
   public:
    TerrainCorrectionExecuter() { std::cout << __FUNCTION__ << std::endl; };

    int Execute() override {
        std::cout << "Executing TC exec test" << std::endl;

        alus::Dataset input(this->input_dataset_name_);
        alus::Dataset dem(this->metadata_folder_path_ + "/srtm_41_01.tif");

        alus::TerrainCorrection terrain_correction(std::move(input), std::move(dem));

        this->ReadOrbitStateVectors((this->metadata_folder_path_ + "/orbit_state_vectors.txt").c_str(), terrain_correction);
        terrain_correction.ExecuteTerrainCorrection(output_file_name_.c_str(), tile_width_, tile_height_);
        return 0;
    }
    [[nodiscard]] RasterDimension CalculateInputTileFrom(RasterDimension output) const override { return output; }
    void SetInputs(const std::string& input_dataset, const std::string& metadata_path) override {
        input_dataset_name_ = input_dataset;
        metadata_folder_path_ = metadata_path;
    }

    void SetTileSize(size_t width, size_t height) override {
        tile_width_ = width;
        tile_height_ = height;
        (void)width;   // Will be used once TC implementation is done. Currently silencing compiler warnings.
        (void)height;  // Will be used once TC implementation is done. Currently silencing compiler warnings.
    }

    void SetOutputFilename(const std::string& output_name) override {
        output_file_name_ = output_name;
        (void)output_name;  // Will be used once TC implementation is done. Currently silencing compiler warnings.
    }

    ~TerrainCorrectionExecuter() override { std::cout << __FUNCTION__ << std::endl; }

   private:
    std::string input_dataset_name_;
    std::string metadata_folder_path_;
    std::string output_file_name_;
    size_t tile_width_;
    size_t tile_height_;

    void ReadOrbitStateVectors(const char *file_name, alus::TerrainCorrection &terrain_correction) {
        std::ifstream data_stream{file_name};
        if (!data_stream.is_open()) {
            throw std::ios::failure("Range Doppler Terrain Correction test data file not open.");
        }
        int test_data_size;
        data_stream >> test_data_size;
        terrain_correction.metadata_.orbit_state_vectors.clear();

        for (int i = 0; i < test_data_size; i++) {
            std::string utc_string1;
            std::string utc_string2;
            double x_pos, y_pos, z_pos, x_vel, y_vel, z_vel;
            data_stream >> utc_string1 >> utc_string2 >> x_pos >> y_pos >> z_pos >> x_vel >> y_vel >> z_vel;
            utc_string1.append(" ");
            utc_string1.append(utc_string2);
            terrain_correction.metadata_.orbit_state_vectors.emplace_back(
                alus::snapengine::old::Utc(utc_string1), x_pos, y_pos, z_pos, x_vel, y_vel, z_vel);
        }

        data_stream.close();
    }
};
}  // namespace alus

extern "C" {
alus::AlgBond* CreateAlgorithm() { return new alus::TerrainCorrectionExecuter(); }

void DeleteAlgorithm(alus::AlgBond* instance) { delete (alus::TerrainCorrectionExecuter*)instance; }
}
