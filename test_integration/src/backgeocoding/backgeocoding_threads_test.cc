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
#include "backgeocoding_controller.h"

#include <vector>

#include "dataset.h"
#include "gmock/gmock.h"
#include "target_dataset.h"
#include "dem_assistant.h"

namespace alus::tests {

// TODO: This test will receive it's final form once backgeocoding is chained with apply orbit file, split and safe
// reading.
TEST(DISABLED_backgeocoding, ThreadTest) {
    std::vector<std::string> srtm3_files{"./goods/srtm_41_01.tif", "./goods/srtm_41_01.tif"};
    std::shared_ptr<alus::app::DemAssistant> dem_assistant = alus::app::DemAssistant::CreateFormattedSrtm3TilesOnGpuFrom(std::move(srtm3_files));
    /*std::cout << "Controller started." << '\n';
    std::shared_ptr<alus::Dataset<double>> master_input_dataset =
        std::make_shared<alus::Dataset<double>>("/home/erik/snapDebusTests/cohTestIn1_split_Orb.tif");
    std::shared_ptr<alus::Dataset<double>> slave_input_dataset =
        std::make_shared<alus::Dataset<double>>("/home/erik/snapDebusTests/cohTestIn2_split_Orb.tif");

    alus::TargetDatasetParams params;
    params.filename = "/home/erik/snapDebusTests/alusTest.tif";
    params.band_count = 4;
    params.driver = master_input_dataset->GetGdalDataset()->GetDriver();
    params.dimension = master_input_dataset->GetRasterDimensions();
    params.transform = master_input_dataset->GetTransform();
    params.projectionRef = master_input_dataset->GetGdalDataset()->GetProjectionRef();

    std::shared_ptr<alus::TargetDataset<float>> output_dataset = std::make_shared<alus::TargetDataset<float>>(params);

    alus::backgeocoding::BackgeocodingController controller(master_input_dataset, slave_input_dataset,
                                                            output_dataset, "./goods/master_metadata.dim",
    "./goods/slave_metadata.dim"); controller.PrepareToCompute(); controller.DoWork();*/

    std::shared_ptr<alus::Dataset<double>> master_input_dataset =
        std::make_shared<alus::Dataset<double>>("/home/erik/snapDebusTests/georgTestMaster_Orb.tif");
    std::shared_ptr<alus::Dataset<double>> slave_input_dataset =
        std::make_shared<alus::Dataset<double>>("/home/erik/snapDebusTests/georgTestSlave_Orb.tif");

    alus::TargetDatasetParams params;
    params.filename = "/home/erik/snapDebusTests/alusTest2.tif";
    params.band_count = 4;
    params.driver = master_input_dataset->GetGdalDataset()->GetDriver();
    params.dimension = master_input_dataset->GetRasterDimensions();
    params.transform = master_input_dataset->GetTransform();
    params.projectionRef = master_input_dataset->GetGdalDataset()->GetProjectionRef();

    std::shared_ptr<alus::TargetDataset<float>> output_dataset = std::make_shared<alus::TargetDataset<float>>(params);

    alus::backgeocoding::BackgeocodingController controller(master_input_dataset, slave_input_dataset, output_dataset,
                                                            "/home/erik/snapDebusTests/georgTestMaster_Orb.dim",
                                                            "/home/erik/snapDebusTests/georgTestSlave_Orb.dim");
    dem_assistant->GetSrtm3Manager()->HostToDevice();
    controller.PrepareToCompute(dem_assistant->GetEgm96Manager()->GetDeviceValues(),
                                {dem_assistant->GetSrtm3Manager()->GetSrtmBuffersInfo(),
                                 dem_assistant->GetSrtm3Manager()->GetDeviceSrtm3TilesCount()});
    controller.DoWork();
}

// TODO: Bring this back with metadata and perhaps integrate into the full test above. It does not hurt to follow the
// TIF standard.
/*TEST(backgeocoding, nodata) {
    std::unique_ptr<alus::Dataset<double>> test_dataset =
        std::make_unique<alus::Dataset<double>>("/home/erik/snapDebusTests/malbolgeTest.tif");
    alus::Rectangle area{4700, 21430, 100, 100};
    std::vector<double> results(10000);

    test_dataset->ReadRectangle(area, 3, results.data());
    double no_data = test_dataset->GetNoDataValue(3);
    std::cout << std::fixed << std::setprecision(14);
    std::cout << "No data value is: " << no_data << std::endl;
    for (int i = 0; i < 20; i++) {
        std::cout << i << ") " << results.at(i) << std::endl;
    }
}*/

}  // namespace alus::tests
