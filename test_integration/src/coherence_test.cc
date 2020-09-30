#include <fstream>

#include "gtest/gtest.h"

// on ubuntu 18.04
// sudo apt install libssl-dev
#include <openssl/md5.h>
#include <sstream>

#include <boost/filesystem.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include "coh_tiles_generator.h"
#include "coherence_calc.h"
#include "gdal_tile_reader.h"
#include "gdal_tile_writer.h"
#include "meta_data.h"
#include "meta_data_node_names.h"
#include "pugixml_meta_data_reader.h"
#include "tf_algorithm_runner.h"

namespace {

std::string Md5FromFile(const std::string& path) {
    unsigned char result[MD5_DIGEST_LENGTH];
    boost::iostreams::mapped_file_source src(path);
    MD5((unsigned char*)src.data(), src.size(), result);
    std::ostringstream sout;
    sout << std::hex << std::setfill('0');
    for (auto c : result) sout << std::setw(2) << (int)c;
    return sout.str();
}

class CoherenceIntegrationTest : public ::testing::Test {
   public:
    CoherenceIntegrationTest() = default;

   protected:
    boost::filesystem::path file_name_in_{"coherence/4_bands.tif"};
    boost::filesystem::path file_name_out_{"4_bands_coh.tif"};
    std::string expected_md5_{"377f197761b36be10c8551c01ac38c62"};

    void SetUp() override { boost::filesystem::remove(file_name_out_); }

    void TearDown() override { boost::filesystem::remove(file_name_out_); }
};

TEST_F(CoherenceIntegrationTest, single_burst_data_2018) {
    {  // ARTIFICAL SCOPE TO FORCE DESTRUCTORS
        constexpr int SRP_NUMBER_POINTS{501};
        constexpr int SRP_POLYNOMIAL_DEGREE{5};
        constexpr bool SUBTRACT_FLAT_EARTH{true};
        constexpr int COH_WIN_RG{15};
        constexpr int COH_WIN_AZ{5};
        // orbit interpolation degree
        constexpr int ORBIT_DEGREE{3};
        // calculation tile size x and y dimension
        constexpr int TILE_X{2675};
        constexpr int TILE_Y{1503};

        const char* FILE_NAME_IA = "coherence/incident_angle.img";
        std::vector<int> band_map_ia{1};
        int band_count_ia = 1;
        alus::GdalTileReader ia_data_reader{FILE_NAME_IA, band_map_ia, band_count_ia, false};
        // small dataset as single tile
        alus::Tile incidence_angle_data_set{ia_data_reader.GetBandXSize() - 1,
                                            ia_data_reader.GetBandYSize() - 1,
                                            ia_data_reader.GetBandXMin(),
                                            ia_data_reader.GetBandYMin()};
        ia_data_reader.ReadTile(incidence_angle_data_set);
        alus::snapengine::PugixmlMetaDataReader xml_reader{
            "coherence/S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_split_Orb_Stack.dim"};
        auto master_root = xml_reader.GetElement(alus::snapengine::MetaDataNodeNames::ABSTRACT_METADATA_ROOT);
        auto slave_root =
            xml_reader.GetElement(alus::snapengine::MetaDataNodeNames::SLAVE_METADATA_ROOT).GetElements().at(0);

        alus::MetaData meta_master{&ia_data_reader, master_root, ORBIT_DEGREE};
        alus::MetaData meta_slave{&ia_data_reader, *slave_root, ORBIT_DEGREE};

        ASSERT_TRUE(boost::filesystem::exists(file_name_in_));
        ASSERT_FALSE(boost::filesystem::exists(file_name_out_));

        // todo:check if bandmap works correctly (e.g if input has 8 bands and we use 1,2,5,6)
        // todo:need some better thought through logic to map inputs from gdal
        std::vector<int> band_map{1, 2, 3, 4};
        std::vector<int> band_map_out{1};
        // might want to take count from coherence?
        int band_count_in = 4;
        int band_count_out = 1;

        alus::GdalTileReader coh_data_reader{file_name_in_.c_str(), band_map, band_count_in, true};
        alus::GdalTileWriter coh_data_writer{file_name_out_.c_str(),
                                             band_map_out,
                                             band_count_out,
                                             coh_data_reader.GetBandXSize(),
                                             coh_data_reader.GetBandYSize(),
                                             coh_data_reader.GetBandXMin(),
                                             coh_data_reader.GetBandYMin(),
                                             coh_data_reader.GetGeoTransform(),
                                             coh_data_reader.GetDataProjection()};
        alus::CohTilesGenerator tiles_generator{
            coh_data_reader.GetBandXSize(), coh_data_reader.GetBandYSize(), TILE_X, TILE_Y, COH_WIN_RG, COH_WIN_RG};
        alus::Coh coherence{SRP_NUMBER_POINTS,
                            SRP_POLYNOMIAL_DEGREE,
                            SUBTRACT_FLAT_EARTH,
                            COH_WIN_RG,
                            COH_WIN_AZ,
                            TILE_X,
                            TILE_Y,
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
    }  // CLOSE ARTIFICAL SCOPE FOR DESTRUCTOR ACTIVATION (important for closing dataset)

    ASSERT_TRUE(boost::filesystem::exists(file_name_out_));
    // make sure output file md5 or pixel values are ok
    ASSERT_EQ(expected_md5_, Md5FromFile(file_name_out_.generic_string()));
}

}  // namespace
