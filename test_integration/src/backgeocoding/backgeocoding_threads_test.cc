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
#include "dataset.h"
#include "gmock/gmock.h"
#include "target_dataset.h"

namespace alus::tests {

//TODO: Fully enable and configure it to use our general test files once the metadata is done in.
TEST(backgeocoding, ThreadTest) {
    std::cout << "Controller started." << '\n';
    std::shared_ptr<alus::Dataset<double>> master_input_dataset =
        std::make_unique<alus::Dataset<double>>("/home/erik/snapDebusTests/cohTestIn1_split_Orb.tif");
    std::shared_ptr<alus::Dataset<double>> slave_input_dataset =
        std::make_unique<alus::Dataset<double>>("/home/erik/snapDebusTests/cohTestIn2_split_Orb.tif");

    alus::TargetDatasetParams params;
    params.filename = "/home/erik/snapDebusTests/alusTest.tif";
    params.band_count = 2;
    params.driver = master_input_dataset->GetGdalDataset()->GetDriver();
    params.dimension = master_input_dataset->GetRasterDimensions();
    params.transform = master_input_dataset->GetTransform();
    params.projectionRef = master_input_dataset->GetGdalDataset()->GetProjectionRef();

    std::shared_ptr<alus::TargetDataset<float>> output_dataset = std::make_unique<alus::TargetDataset<float>>(params);

    alus::backgeocoding::BackgeocodingController controller(master_input_dataset, slave_input_dataset,
                                                            output_dataset);
    controller.PrepareToCompute();
    controller.StartWork();
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
