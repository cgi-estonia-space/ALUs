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
#include "gmock/gmock.h"

#include <sstream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include <openssl/md5.h>

#include "band_params.h"
#include "coh_tiles_generator.h"
#include "coh_window.h"
#include "coherence_calc_cuda.h"
#include "cuda_algorithm_runner.h"
#include "gdal_tile_reader.h"
#include "gdal_tile_writer.h"
#include "meta_data.h"
#include "pugixml_meta_data_reader.h"
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata.h"

namespace {

std::string Md5FromFile(const std::string& path) {
    unsigned char result[MD5_DIGEST_LENGTH];
    boost::iostreams::mapped_file_source src(path);
    MD5((unsigned char*)src.data(), src.size(), result);
    std::ostringstream sout;
    sout << std::hex << std::setfill('0');
    for (auto c : result) sout << std::setw(2) << static_cast<int>(c);
    return sout.str();
}

class CoherenceIntegrationTest : public ::testing::Test {
public:
    CoherenceIntegrationTest() = default;

protected:
    boost::filesystem::path file_name_in_{"./goods/coherence/4_bands.tif"};
    boost::filesystem::path file_name_out_{"/tmp/4_bands_cuda_coh.tif"};

    void SetUp() override { boost::filesystem::remove(file_name_out_); }
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
        constexpr int TILE_X_SIZE{2675};
        constexpr int TILE_Y_SIZE{1503};

        const char* file_name_ia = "./goods/coherence/incident_angle.img";
        std::vector<int> band_map_ia{1};
        int band_count_ia = 1;
        alus::coherence_cuda::GdalTileReader ia_data_reader{file_name_ia, band_map_ia, band_count_ia, false};
        // small dataset as single tile
        alus::Tile incidence_angle_data_set{ia_data_reader.GetBandXSize() - 1,
                                                            ia_data_reader.GetBandYSize() - 1,
                                                            ia_data_reader.GetBandXMin(), ia_data_reader.GetBandYMin()};
        ia_data_reader.ReadTile(incidence_angle_data_set);
        alus::snapengine::PugixmlMetaDataReader xml_reader{
            "./goods/coherence/S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_split_Orb_Stack"
            ".dim"};
        auto master_root = xml_reader.Read(alus::snapengine::AbstractMetadata::ABSTRACT_METADATA_ROOT);
        auto slave_root = xml_reader.Read(alus::snapengine::AbstractMetadata::SLAVE_METADATA_ROOT)->GetElements().at(0);

        alus::coherence_cuda::MetaData meta_master{&ia_data_reader, master_root, ORBIT_DEGREE};
        alus::coherence_cuda::MetaData meta_slave{&ia_data_reader, slave_root, ORBIT_DEGREE};

        ASSERT_TRUE(boost::filesystem::exists(file_name_in_));
        ASSERT_FALSE(boost::filesystem::exists(file_name_out_));

        // todo:check if bandmap works correctly (e.g if input has 8 bands and we use 1,2,5,6)
        // todo:need some better thought through logic to map inputs from gdal
        std::vector<int> band_map{1, 2, 3, 4};
        std::vector<int> band_map_out{1};
        // might want to take count from coherence?
        int band_count_in = 4;
        int band_count_out = 1;

        alus::coherence_cuda::GdalTileReader coh_data_reader{file_name_in_.c_str(), band_map, band_count_in, true};

        alus::BandParams band_params{band_map_out,
                                                     band_count_out,
                                                     coh_data_reader.GetBandXSize(),
                                                     coh_data_reader.GetBandYSize(),
                                                     coh_data_reader.GetBandXMin(),
                                                     coh_data_reader.GetBandYMin()};

        alus::coherence_cuda::GdalTileWriter coh_data_writer{
            file_name_out_.c_str(), band_params, coh_data_reader.GetGeoTransform(), coh_data_reader.GetDataProjection()};

        alus::coherence_cuda::CohTilesGenerator tiles_generator{coh_data_reader.GetBandXSize(),
                                                                coh_data_reader.GetBandYSize(),
                                                                TILE_X_SIZE,
                                                                TILE_Y_SIZE,
                                                                COH_WIN_RG,
                                                                COH_WIN_AZ};

        alus::coherence_cuda::CohWindow coh_window{COH_WIN_RG, COH_WIN_AZ};
        alus::coherence_cuda::CohCuda coherence{SRP_NUMBER_POINTS,
                                                SRP_POLYNOMIAL_DEGREE,
                                                SUBTRACT_FLAT_EARTH,
                                                coh_window,
                                                ORBIT_DEGREE,
                                                meta_master,
                                                meta_slave};

        //    // run the algorithm
        alus::coherence_cuda::CUDAAlgorithmRunner cuda_algo_runner{&coh_data_reader, &coh_data_writer, &tiles_generator,
                                                                   &coherence};
        cuda_algo_runner.Run();
    }  // CLOSE ARTIFICAL SCOPE FOR DESTRUCTOR ACTIVATION (important for closing dataset)

    ASSERT_TRUE(boost::filesystem::exists(file_name_out_));
    // make sure output file md5 or pixel values are ok
    std::string expected_md5{"7b30dd7e3c04cf819fe288490b0f1e3e"};
    std::string expected_md5_2 {"79d887102ec90671354f43778070898f"};
    auto actual_md5 = Md5FromFile(file_name_out_.generic_string());

    if(actual_md5 == expected_md5 || actual_md5 == expected_md5_2) return;

    // release and debug builds can differ due to floating point math
    // unfortunately this needs a bigger refactor in the future
    ASSERT_TRUE(false) << " actual md5" << actual_md5 << " expected: " << expected_md5 << ", "
                       << expected_md5_2;

}

}  // namespace
