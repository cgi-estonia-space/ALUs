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

#include <memory>

#include "gmock/gmock.h"

#include "topsar_split.h"
#include "i_meta_data_reader.h"
#include "pugixml_meta_data_reader.h"
#include "product.h"
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata.h"

#include "apply_orbit_file_op.h"
#include "topsar_split.h"
#include "snap-core/util/system_utils.h"
#include "target_dataset.h"
#include "c16_dataset.h"

namespace {

TEST(DISABLED_topsar_split, subswaths){

    alus::topsarsplit::TopsarSplit splitter("/home/erik/snapDebusTests/S1A_IW_SLC__1SDV_20210413T043427_20210413T043455_037427_046960_3797.SAFE", "IW1", "VV");
    splitter.initialize();
}

TEST(DISABLED_topsar_split, s1utils){
    alus::snapengine::SystemUtils::SetAuxDataPath("/home/erik/orbitFiles/POEORB/");
    std::string subswath_name = "IW1";
    std::string polarisation = "VV";
    std::string slave_file = "/home/erik/snapDebusTests/BEIRUT/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE";
    alus::topsarsplit::TopsarSplit split_slave(slave_file, subswath_name, polarisation);
    split_slave.initialize();
    std::shared_ptr<alus::C16Dataset<double>> master_reader = split_slave.GetPixelReader();
    alus::s1tbx::ApplyOrbitFileOp orbit_file_slave(split_slave.GetTargetProduct(), true);
    orbit_file_slave.Initialize();

    alus::s1tbx::Sentinel1Utils slave_utils(split_slave.GetTargetProduct());
    slave_utils.ComputeDopplerRate();
    slave_utils.ComputeReferenceTime();
    slave_utils.subswath_.at(0)->HostToDevice();
    slave_utils.HostToDevice();

    //TODO: now compare slave_utils to what we would get from snap.
}

}
