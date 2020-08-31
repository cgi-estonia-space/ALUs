#include <iostream>

#include "alg_bond.h"
#include "coh_tiles_generator.h"
#include "coherence_calc.h"
#include "gdal_tile_reader.h"
#include "gdal_tile_writer.h"
#include "meta_data.h"
#include "tf_algorithm_runner.h"

namespace alus {
class CoherenceExecuter : public AlgBond {
   public:
    CoherenceExecuter() { std::cout << __FUNCTION__ << std::endl; };

    int Execute() override {
        std::cout << "Executing Coherence test" << std::endl;

        constexpr int SRP_NUMBER_POINTS{501};
        constexpr int SRP_POLYNOMIAL_DEGREE{5};
        constexpr bool SUBTRACT_FLAT_EARTH{true};
        constexpr int COH_WIN_RG{15};
        constexpr int COH_WIN_AZ{5};
        // orbit interpolation degree
        constexpr int ORBIT_DEGREE{3};

        std::vector<double> time_master = {56898.961993, 56899.961993, 56900.961993, 56901.961993, 56902.961993,
                                           56903.961993, 56904.961993, 56905.961993, 56906.961993, 56907.961993,
                                           56908.961993, 56909.961993, 56910.961993, 56911.961993, 56912.961993,
                                           56913.961993, 56914.961993, 56915.961993, 56916.961993, 56917.961993,
                                           56918.961993, 56919.961993, 56920.961993};
        std::vector<double> coeff_x_master = {
            3500544.0355238644, -50184.20231744633, -226.93150051496792, 0.9224466101732173};
        std::vector<double> coeff_y_master = {
            1250765.4708142395, -42396.32190925344, -33.44925512676017, 0.9022354812477725};
        std::vector<double> coeff_z_master = {
            6010846.398902166, 37960.50551962845, -339.08156203292407, -0.7101358745421649};

        std::vector<double> time_slave = {56899.695959, 56900.695959, 56901.695959, 56902.695959, 56903.695959,
                                          56904.695959, 56905.695959, 56906.695959, 56907.695959, 56908.695959,
                                          56909.695959, 56910.695959, 56911.695959, 56912.695959, 56913.695959,
                                          56914.695959, 56915.695959, 56916.695959, 56917.695959, 56918.695959,
                                          56919.695959, 56920.695959, 56921.695959};
        std::vector<double> coeff_x_slave = {
            3500485.5070849694, -50184.73385613251, -226.92776008788366, 0.9222755703303914};
        std::vector<double> coeff_y_slave = {
            1250766.9302500854, -42396.05110474422, -33.449026719665916, 0.9021607859394872};
        std::vector<double> coeff_z_slave = {
            6010879.846442485, 37960.048844808814, -339.0831543365503, -0.7100311931717342};

        auto const FILE_NAME_IA = aux_location_ + "/incident_angle.img";
        std::vector<int> band_map_ia{1};
        int band_count_ia = 1;
        alus::GdalTileReader ia_data_reader{FILE_NAME_IA, band_map_ia, band_count_ia, false};
        // small dataset as single tile
        alus::Tile incidence_angle_data_set{ia_data_reader.GetBandXSize() - 1,
                                            ia_data_reader.GetBandYSize() - 1,
                                            ia_data_reader.GetBandXMin(),
                                            ia_data_reader.GetBandYMin()};
        ia_data_reader.ReadTile(incidence_angle_data_set);

        alus::MetaData meta_master{
            &ia_data_reader, 56908.961993, time_master, coeff_x_master, coeff_y_master, coeff_z_master};
        alus::MetaData meta_slave{
            &ia_data_reader, 56909.695959, time_slave, coeff_x_slave, coeff_y_slave, coeff_z_slave};

        // todo:check if bandmap works correctly (e.g if input has 8 bands and we use 1,2,5,6)
        // todo:need some better thought through logic to map inputs from gdal
        std::vector<int> band_map{1, 2, 3, 4};
        std::vector<int> band_map_out{1};
        // might want to take count from coherence?
        int band_count_in = 4;
        int band_count_out = 1;

        alus::GdalTileReader coh_data_reader{input_name_, band_map, band_count_in, true};
        alus::GdalTileWriter coh_data_writer{output_name_,
                                             band_map_out,
                                             band_count_out,
                                             coh_data_reader.GetBandXSize(),
                                             coh_data_reader.GetBandYSize(),
                                             coh_data_reader.GetBandXMin(),
                                             coh_data_reader.GetBandYMin(),
                                             coh_data_reader.GetGeoTransform(),
                                             coh_data_reader.GetDataProjection()};
        alus::CohTilesGenerator tiles_generator{coh_data_reader.GetBandXSize(),
                                                coh_data_reader.GetBandYSize(),
                                                tile_width_,
                                                tile_height_,
                                                COH_WIN_RG,
                                                COH_WIN_RG};
        alus::Coh coherence{SRP_NUMBER_POINTS,
                            SRP_POLYNOMIAL_DEGREE,
                            SUBTRACT_FLAT_EARTH,
                            COH_WIN_RG,
                            COH_WIN_AZ,
                            tile_width_,
                            tile_height_,
                            ORBIT_DEGREE,
                            meta_master,
                            meta_slave};

        // create session for tensorflow
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        auto options = tensorflow::SessionOptions();
        options.config.mutable_gpu_options()->set_allow_growth(true);
        tensorflow::ClientSession session(root, options);

        // run the algorithm
        alus::TFAlgorithmRunner tf_algo_runner{
            &coh_data_reader, &coh_data_writer, &tiles_generator, &coherence, &session, &root};
        tf_algo_runner.Run();

        return 0;
    }
    [[nodiscard]] RasterDimension CalculateInputTileFrom(RasterDimension output) const override { return output; }

    void SetInputs(const std::string& input_dataset, const std::string& metadata_path) override {
        input_name_ = input_dataset;
        aux_location_ = metadata_path;
    }

    void SetTileSize(size_t width, size_t height) override {
        tile_width_ = width;
        tile_height_ = height;
    }

    void SetOutputFilename(const std::string& output_name) override { output_name_ = output_name; }

    ~CoherenceExecuter() override { std::cout << __FUNCTION__ << std::endl; };

   private:
    std::string input_name_{};
    std::string output_name_{};
    std::string aux_location_{};
    size_t tile_width_{};
    size_t tile_height_{};
};
}  // namespace alus

extern "C" {
alus::AlgBond* CreateAlgorithm() { return new alus::CoherenceExecuter(); }

void DeleteAlgorithm(alus::AlgBond* instance) { delete (alus::CoherenceExecuter*)instance; }
}
