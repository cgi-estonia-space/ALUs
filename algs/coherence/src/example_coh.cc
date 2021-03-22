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

#include "coh_tiles_generator.h"
#include "coherence_calc.h"
#include "gdal_tile_reader.h"
#include "gdal_tile_writer.h"
#include "meta_data.h"
#include "pugixml_meta_data_reader.h"
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "tf_algorithm_runner.h"

namespace alus {
// todo:user should be able to alter these in the future
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

}  // namespace alus
int main() {
    const char* FILE_NAME_IA = "incident_angle.img";
    std::vector<int> band_map_ia{1};
    int band_count_ia = 1;
    alus::GdalTileReader ia_data_reader{FILE_NAME_IA, band_map_ia, band_count_ia, false};
    // small dataset as single tile
    alus::Tile incidence_angle_data_set{ia_data_reader.GetBandXSize() - 1, ia_data_reader.GetBandYSize() - 1,
                                        ia_data_reader.GetBandXMin(), ia_data_reader.GetBandYMin()};
    ia_data_reader.ReadTile(incidence_angle_data_set);

    alus::snapengine::PugixmlMetaDataReader xml_reader{
        "S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_split_Orb_Stack.dim"};
    auto master_root = xml_reader.Read(alus::snapengine::AbstractMetadata::ABSTRACT_METADATA_ROOT);
    auto slave_root = xml_reader.Read(alus::snapengine::AbstractMetadata::SLAVE_METADATA_ROOT)->GetElements().at(0);

    alus::MetaData meta_master{&ia_data_reader, master_root, alus::ORBIT_DEGREE};
    alus::MetaData meta_slave{&ia_data_reader, slave_root, alus::ORBIT_DEGREE};

    const char* FILE_NAME_IN = "4_bands.tif";  // this has minimal needed input bands (master/slave real and imaginary)
    const char* FILE_NAME_OUT = "4_bands_coh.tif";

    // todo:check if bandmap works correctly (e.g if input has 8 bands and we use 1,2,5,6)
    // todo:need some better thought through logic to map inputs from gdal
    std::vector<int> band_map{1, 2, 3, 4};
    std::vector<int> band_map_out{1};
    // might want to take count from coherence?
    int band_count_in = 4;
    int band_count_out = 1;

    alus::GdalTileReader coh_data_reader{FILE_NAME_IN, band_map, band_count_in, true};
    alus::GdalTileWriter coh_data_writer{FILE_NAME_OUT,
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
                                            alus::TILE_X,
                                            alus::TILE_Y,
                                            alus::COH_WIN_RG,
                                            alus::COH_WIN_RG};
    alus::Coh coherence{alus::SRP_NUMBER_POINTS,
                        alus::SRP_POLYNOMIAL_DEGREE,
                        alus::SUBTRACT_FLAT_EARTH,
                        alus::COH_WIN_RG,
                        alus::COH_WIN_AZ,
                        alus::TILE_X,
                        alus::TILE_Y,
                        alus::ORBIT_DEGREE,
                        meta_master,
                        meta_slave};

    // create session for tensorflow
    tensorflow::Scope root = tensorflow::Scope::NewRootScope();
    auto options = tensorflow::SessionOptions();
    options.config.mutable_gpu_options()->set_allow_growth(true);
    tensorflow::ClientSession session(root, options);

    // run the algorithm
    alus::TFAlgorithmRunner tf_algo_runner{&coh_data_reader, &coh_data_writer, &tiles_generator,
                                           &coherence,       &session,         &root};
    tf_algo_runner.Run();
}
