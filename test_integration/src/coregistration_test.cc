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

#include "coregistration_controller.h"
#include "dem_assistant.h"

namespace {

TEST(tcoregistration, all3) {
    std::vector<std::string> srtm3_files{"./goods/srtm_43_06.tif", "./goods/srtm_44_06.tif"};
    std::shared_ptr<alus::app::DemAssistant> dem_assistant =
        alus::app::DemAssistant::CreateFormattedSrtm3TilesOnGpuFrom(std::move(srtm3_files));

    alus::coregistration::Coregistration cor("/home/erik/orbitFiles/POEORB/");
    cor.Initialize(
        "/home/erik/snapDebusTests/BEIRUT/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA.SAFE",
        "/home/erik/snapDebusTests/BEIRUT/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE",
        "/home/erik/snapDebusTests/BEIRUT/yes_kaboom_rico.tif", "IW1", "VV");
    dem_assistant->GetSrtm3Manager()->HostToDevice();
    cor.DoWork(dem_assistant->GetEgm96Manager()->GetDeviceValues(),
               {dem_assistant->GetSrtm3Manager()->GetSrtmBuffersInfo(),
                dem_assistant->GetSrtm3Manager()->GetDeviceSrtm3TilesCount()});
}

}  // namespace
