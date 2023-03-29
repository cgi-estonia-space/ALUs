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

#include "alus_log.h"
#include "band_params.h"
#include "coh_tiles_generator.h"
#include "coh_window.h"
#include "coherence_calc_cuda.h"
#include "cuda_algorithm_runner.h"
#include "gdal_tile_reader.h"
#include "gdal_tile_writer.h"
#include "meta_data.h"
#include "pugixml_meta_data_reader.h"
#include "snap-engine-utilities/engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "test_utils.h"

namespace {

class CoherenceIntegrationTest : public ::testing::Test {
public:
    CoherenceIntegrationTest() = default;

protected:
    boost::filesystem::path file_name_in_{"./goods/coherence/4_bands.tif"};
    void RunSingleBurstData2018Test(const std::string& file_name_out, int tile_x_size, int tile_y_size,
                                    std::string& band_hash) {
        {
            constexpr int SRP_NUMBER_POINTS{501};
            constexpr int SRP_POLYNOMIAL_DEGREE{5};
            constexpr bool SUBTRACT_FLAT_EARTH{false};  // TODO, sfep no tested, MetaData root -> Product needs to be
                                                        // implemented for per burst metadata in coherence
            constexpr int COH_WIN_RG{15};
            constexpr int COH_WIN_AZ{5};
            // orbit interpolation degree
            constexpr int ORBIT_DEGREE{3};

            const char* file_name_ia = "./goods/coherence/incident_angle.img";
            alus::coherence_cuda::GdalTileReader ia_data_reader{file_name_ia};
            alus::snapengine::PugixmlMetaDataReader xml_reader{
                "./goods/coherence/"
                "S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_split_Orb_Stack.dim"};
            auto master_root = xml_reader.Read(alus::snapengine::AbstractMetadata::ABSTRACT_METADATA_ROOT);
            auto slave_root =
                xml_reader.Read(alus::snapengine::AbstractMetadata::SLAVE_METADATA_ROOT)->GetElements().at(0);

            const int ia_width = ia_data_reader.GetBandXSize();
            const int ia_height = ia_data_reader.GetBandYSize();
            auto avg_incidence_angle = static_cast<float>(ia_data_reader.GetValueAtXy(ia_width / 2, ia_height / 2));
            bool is_near_range_left = ia_data_reader.GetValueAtXy(0, 0) < ia_data_reader.GetValueAtXy(ia_width - 1, 0);

            alus::coherence_cuda::MetaData meta_master{is_near_range_left, master_root, ORBIT_DEGREE,
                                                       avg_incidence_angle};
            alus::coherence_cuda::MetaData meta_slave{is_near_range_left, slave_root, ORBIT_DEGREE,
                                                      avg_incidence_angle};

            ASSERT_TRUE(boost::filesystem::exists(file_name_in_));
            boost::filesystem::remove(file_name_out);

            std::vector<int> band_map_out{1};
            // might want to take count from coherence?
            int band_count_out = 1;

            alus::coherence_cuda::GdalTileReader coh_data_reader{file_name_in_.c_str()};

            alus::BandParams band_params{band_map_out,
                                         band_count_out,
                                         coh_data_reader.GetBandXSize(),
                                         coh_data_reader.GetBandYSize(),
                                         coh_data_reader.GetBandXMin(),
                                         coh_data_reader.GetBandYMin()};

            alus::coherence_cuda::GdalTileWriter coh_data_writer{
                file_name_out, band_params, coh_data_reader.GetGeoTransform(), coh_data_reader.GetDataProjection()};

            int lines_per_burst = 1503;
            alus::coherence_cuda::CohTilesGenerator tiles_generator{coh_data_reader.GetBandXSize(),
                                                                    coh_data_reader.GetBandYSize(),
                                                                    tile_x_size,
                                                                    tile_y_size,
                                                                    COH_WIN_RG,
                                                                    COH_WIN_AZ,
                                                                    lines_per_burst};

            alus::coherence_cuda::CohWindow coh_window{COH_WIN_RG, COH_WIN_AZ};
            alus::coherence_cuda::CohCuda coherence{SRP_NUMBER_POINTS, SRP_POLYNOMIAL_DEGREE, SUBTRACT_FLAT_EARTH,
                                                    coh_window,        ORBIT_DEGREE,          meta_master,
                                                    meta_slave};

            //    // run the algorithm
            alus::coherence_cuda::CUDAAlgorithmRunner cuda_algo_runner{&coh_data_reader, &coh_data_writer,
                                                                       &tiles_generator, &coherence};
            cuda_algo_runner.Run();
        }  // CLOSE ARTIFICAL SCOPE FOR DESTRUCTOR ACTIVATION (important for closing dataset)

        ASSERT_TRUE(boost::filesystem::exists(file_name_out));
        band_hash = alus::utils::test::HashFromBand(file_name_out);
    }
};

TEST_F(CoherenceIntegrationTest, singleBurstData2018) {
    const std::string file_out = "/tmp/4_bands_cuda_coh.tif";
    boost::filesystem::remove(file_out);

    const std::vector<int> tile_x = {227, 229, 233, 239, 241};
    const std::vector<int> tile_y = {331, 337, 347, 349, 353};

    std::string prev_band_hash;
    const int test_tile_count{5};
    for (int i = 0; i < test_tile_count; i++) {
        std::string band_hash;
        RunSingleBurstData2018Test(file_out, tile_x.at(i), tile_y.at(i), band_hash);
        if (i > 0) {
            EXPECT_EQ(prev_band_hash, band_hash) << "index = " << i;
        }
        CHECK_CUDA_ERR(cudaGetLastError());
        CHECK_CUDA_ERR(cudaDeviceSynchronize());
        CHECK_CUDA_ERR(cudaDeviceReset());  // for cuda-memcheck --leak-check full
        prev_band_hash = band_hash;
    }

    const std::string actual_hash = prev_band_hash;
    const std::string expected_hash1 = "38f8960cf80ef2a8";
    const std::string expected_hash2 = "ae3a8961ab05ddb3";
    const std::string expected_hash3 = "56e1b4de42de8cb9";

    if (actual_hash == expected_hash1 || actual_hash == expected_hash2 || actual_hash == expected_hash3) return;

    // release and debug builds can differ due to floating point math
    // unfortunately this needs a bigger refactor in the future
    ASSERT_TRUE(false) << " actual md5: " << actual_hash << " expected: " << expected_hash1 << " or " << expected_hash2;
}

}  // namespace
