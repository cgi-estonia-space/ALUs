
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <gdal_priv.h>

#include "gmock/gmock.h"

#include "filter_bank.h"

// Symbols from linked "filter_banks_gabor_6or_4f.o"
extern char _binary_filter_banks_gabor_6or_4f_bin_start[];
extern char _binary_filter_banks_gabor_6or_4f_bin_end[];

namespace {

using ::testing::ContainerEq;
using ::testing::DoubleNear;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::Gt;
using ::testing::Lt;
using ::testing::Pointwise;
using ::testing::SizeIs;

const std::array<float, 4> FREQUENCIES_4_ELEMENT_SAMPLE{0.125f, 0.375f, 0.625f, 0.875f};
const std::array<float, 6> ORIENTATIONS_6_ELEMENT_SAMPLE{0.0f,        0.5235988f,  1.04719758f,
                                                         1.57079637f, 2.09439516f, 2.61799383f};

[[maybe_unused]] void saveFiltersAsImages(const std::vector<std::vector<float>>& filter_bank, std::string_view folder) {
    GDALDriver* output_driver = GetGDALDriverManager()->GetDriverByName("GTiff");

    size_t dimension_last_bank{static_cast<size_t>(std::sqrt(filter_bank.at(0).size()))};
    std::vector<float> buffer{};
    for (size_t i{0}; i < filter_bank.size(); i++) {
        const auto& b = filter_bank.at(i);
        const size_t filter_dimension = static_cast<size_t>(std::sqrt(b.size()));
        std::cout << filter_dimension << std::endl;

        if (filter_dimension != dimension_last_bank || i == filter_bank.size() - 1) {
            if (i == filter_bank.size() - 1) {
                std::copy(b.cbegin(), b.cend(), std::back_inserter(buffer));
            }
            const size_t filters = buffer.size() / (dimension_last_bank * dimension_last_bank);
            std::string filepath = std::string(folder) + "/gabor_bank_" + std::to_string(dimension_last_bank) + "_" +
                                   std::to_string(dimension_last_bank) + ".tif";
            std::cout << "Saving " << filters << " filters to " << std::endl;
            auto* output_dataset =
                output_driver->Create(filepath.c_str(), dimension_last_bank, filters * dimension_last_bank, 1,
                                      GDALDataType::GDT_Float32, nullptr);
            auto err = output_dataset->GetRasterBand(1)->RasterIO(
                GF_Write, 0, 0, dimension_last_bank, filters * dimension_last_bank, buffer.data(), dimension_last_bank,
                filters * dimension_last_bank, GDALDataType::GDT_Float32, 0, 0, nullptr);
            if (err != CE_None) {
                std::cerr << "Saving filter resulted in error -" << CPLGetLastErrorMsg() << std::endl;
            }

            GDALClose(output_dataset);

            if (i == filter_bank.size() - 1) {
                break;
            }

            dimension_last_bank = filter_dimension;
            buffer = {};
        }

        std::copy(b.cbegin(), b.cend(), std::back_inserter(buffer));
    }
}

TEST(FilterBank, GeneratesFrequenciesCountAndRangeCorrectly) {
    for (size_t freq = 1; freq < 10; freq++) {
        auto&& list = alus::featurextractiongabor::GenerateFrequencies(freq);
        ASSERT_THAT(list.size(), Eq(freq));
        ASSERT_TRUE(std::is_sorted(list.cbegin(), list.cend()) &&
                    std::adjacent_find(list.cbegin(), list.cend()) == list.cend());
        ASSERT_THAT(list.front(), Gt(0));
        ASSERT_THAT(list.back(), Lt(1));
    }
}

TEST(FilterBank, GeneratesFrequenciesInEvenSteps) {
    const auto count{10};
    auto&& list = alus::featurextractiongabor::GenerateFrequencies(count);
    const auto expected_step = list.at(1) - list.at(0);
    for (auto i{1}; i < count; i++) {
        ASSERT_THAT(list.at(i) - list.at(i - 1), FloatNear(expected_step, 0.001));
    }
}

TEST(FilterBank, Generates4CorrectFrequencies) {
    auto&& list = alus::featurextractiongabor::GenerateFrequencies(4);
    ASSERT_THAT(list, ElementsAreArray(FREQUENCIES_4_ELEMENT_SAMPLE.cbegin(), FREQUENCIES_4_ELEMENT_SAMPLE.cend()));
}

TEST(FilterBank, GeneratesOrientationsCountAndRangeCorrectly) {
    for (size_t oc = 1; oc < 10; oc++) {
        auto&& list = alus::featurextractiongabor::GenerateOrientations(oc);
        ASSERT_THAT(list.size(), Eq(oc));
        ASSERT_TRUE(std::is_sorted(list.cbegin(), list.cend()) &&
                    std::adjacent_find(list.cbegin(), list.cend()) == list.cend());
        ASSERT_THAT(list.front(), FloatEq(0));
        if (oc > 1) {
            ASSERT_THAT(list.back(), Lt(M_PIf32));
        }
    }
}

TEST(FilterBank, GeneratesOrientationsInEvenSteps) {
    const auto count{10};
    auto&& list = alus::featurextractiongabor::GenerateOrientations(count);
    const auto expected_step = list.at(1) - list.at(0);
    for (auto i{1}; i < count; i++) {
        ASSERT_THAT(list.at(i) - list.at(i - 1), FloatNear(expected_step, 0.001));
    }
}

TEST(FilterBank, Generates6CorrectOrientations) {
    auto&& list = alus::featurextractiongabor::GenerateOrientations(6);
    ASSERT_THAT(list, ElementsAreArray(ORIENTATIONS_6_ELEMENT_SAMPLE.cbegin(), ORIENTATIONS_6_ELEMENT_SAMPLE.cend()));
}

TEST(FilterBank, ComputesFilterDimensionSeedCorrectly) {
    const std::vector<float> dummy_sigmas{4.48f, 1.49333334f, 0.896f, 0.640000045f};
    const std::vector<size_t> expected_filter_dimension_seeds{13, 4, 3, 2};
    for (size_t i = 0; i < dummy_sigmas.size(); i++) {
        ASSERT_THAT(alus::featurextractiongabor::ComputeFilterDimensionSeed(dummy_sigmas.at(i)),
                    Eq(expected_filter_dimension_seeds.at(i)));
    }
}

TEST(FilterBank, ComputesFilterDimensionCorrectly) {
    const std::vector<size_t> dummy_filter_dimension_seeds{13, 4, 3, 2};
    const std::vector<size_t> expected_dimensions{27, 9, 7, 5};
    for (size_t i = 0; i < dummy_filter_dimension_seeds.size(); i++) {
        ASSERT_THAT(alus::featurextractiongabor::ComputeFilterDimension(dummy_filter_dimension_seeds.at(i)),
                    Eq(expected_dimensions.at(i)));
    }
}

TEST(FilterBank, ComputesWavelengthCorrectly) {
    const std::vector<float> frequencies{0.125f, 0.375f, 0.625f, 0.875f};
    const std::vector<float> expected{8.0f, 2.66666675f, 1.6f, 1.14285719f};
    const auto& results = alus::featurextractiongabor::ComputeWavelengthsFrom(frequencies);
    ASSERT_THAT(results, Pointwise(FloatNear(1e-8), expected));
}

TEST(FilterBank, ComputesSigmaCorrectly) {
    const std::vector<float> wavelengths{8.0f, 2.66666675f, 1.6f, 1.14285719f};
    const std::vector<float> expected{4.48f, 1.49333334f, 0.896f, 0.640000045f};
    const auto& results = alus::featurextractiongabor::ComputeSigmasFrom(wavelengths);
    ASSERT_THAT(results, Pointwise(FloatNear(1e-8), expected));
}

TEST(FilterBank, ComputeThetaThrowsInvalidArgumentExceptionWhenBufferSizeDoesNotMatchFilterDimensionSeed) {
    float buf[4];
    ASSERT_THROW(alus::featurextractiongabor::ComputeThetaX(0.1f, 1, buf, 4), std::invalid_argument);
    ASSERT_THROW(alus::featurextractiongabor::ComputeThetaY(0.1f, 1, buf, 4), std::invalid_argument);
}

TEST(FilterBank, ComputeThetaAssignsAllValues) {
    {
        std::vector<float> results(7 * 7, NAN);
        alus::featurextractiongabor::ComputeThetaX(0.488f, 3, results.data(), results.size());
        for (auto&& v : results) {
            ASSERT_THAT(std::isnan(v), ::testing::IsFalse());
        }
    }
    {
        std::vector<float> results(27 * 27, NAN);
        alus::featurextractiongabor::ComputeThetaY(15.43f, 13, results.data(), results.size());
        for (auto&& v : results) {
            ASSERT_THAT(std::isnan(v), ::testing::IsFalse());
        }
    }
}

TEST(FilterBank, ComputesThetaCorrectlyForX) {
    {
        const std::vector<float> expected{-2.0f, -1.0f, 0.0f,  1.0f,  2.0f, -2.0f, -1.0f, 0.0f,  1.0f,
                                          2.0f,  -2.0f, -1.0f, 0.0f,  1.0f, 2.0f,  -2.0f, -1.0f, 0.0f,
                                          1.0f,  2.0f,  -2.0f, -1.0f, 0.0f, 1.0f,  2.0f};
        std::vector<float> results(5 * 5);
        alus::featurextractiongabor::ComputeThetaX(0, 2, results.data(), results.size());
        ASSERT_THAT(results, Pointwise(FloatNear(1e-8), expected));
    }
    {
        const std::vector<float> expected{0.732050657f, -0.133974731f, -1.00000012f, -1.86602545f,  -2.732051f,
                                          1.23205066f,  0.366025329f,  -0.50000006f, -1.36602545f,  -2.232051f,
                                          1.73205078f,  0.8660254f,    0.0f,         -0.8660254f,   -1.73205078f,
                                          2.232051f,    1.36602545f,   0.50000006f,  -0.366025329f, -1.23205066f,
                                          2.732051f,    1.86602545f,   1.00000012f,  0.133974731f,  -0.732050657f};
        std::vector<float> results(5 * 5);
        alus::featurextractiongabor::ComputeThetaX(2.61799383f, 2, results.data(), results.size());
        ASSERT_THAT(results, Pointwise(FloatNear(1e-8), expected));
    }
}

TEST(FilterBank, ComputesThetaCorrectlyForY) {
    {
        const std::vector<float> expected{-2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                          -1.0f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  1.0f,  1.0f,  1.0f,
                                          1.0f,  1.0f,  2.0f,  2.0f,  2.0f,  2.0f,  2.0f};
        std::vector<float> results(5 * 5);
        alus::featurextractiongabor::ComputeThetaY(0, 2, results.data(), results.size());
        ASSERT_THAT(results, Pointwise(FloatNear(1e-8), expected));
    }

    {
        const std::vector<float> expected{2.732051f,    1.86602545f,   1.00000012f,  0.133974731f,  -0.732050657f,
                                          2.232051f,    1.36602545f,   0.50000006f,  -0.366025329f, -1.23205066f,
                                          1.73205078f,  0.8660254f,    0.0f,         -0.8660254f,   -1.73205078f,
                                          1.23205066f,  0.366025329f,  -0.50000006f, -1.36602545f,  -2.232051f,
                                          0.732050657f, -0.133974731f, -1.00000012f, -1.86602545f,  -2.732051f};
        std::vector<float> results(5 * 5);
        alus::featurextractiongabor::ComputeThetaY(2.09439516f, 2, results.data(), results.size());
        ASSERT_THAT(results, Pointwise(FloatNear(1e-8), expected));
    }
}

TEST(FilterBank, ComputesFilterBankItemCorrectly) {
    const std::vector<float> theta_x{-2.732051f,   -1.86602545f, -1.0f, -0.133974612f, 0.7320508f,
                                     -2.232051f,   -1.36602545f, -0.5f, 0.3660254f,    1.23205078f,
                                     -1.73205078f, -0.8660254f,  0.0f,  0.8660254f,    1.73205078f,
                                     -1.23205078f, -0.3660254f,  0.5f,  1.36602545f,   2.232051f,
                                     -0.7320508f,  0.133974612f, 1.0f,  1.86602545f,   2.732051f};
    const std::vector<float> theta_y{-0.7320508f,  -1.23205078f, -1.73205078f, -2.232051f,   -2.732051f,
                                     0.133974612f, -0.3660254f,  -0.8660254f,  -1.36602545f, -1.86602545F,
                                     1.0f,         0.5f,         0.0f,         -0.5f,        -1.0f,
                                     1.86602545f,  1.36602545f,  0.8660254f,   0.3660254f,   -0.133974612f,
                                     2.732051f,    2.232051f,    1.73205078f,  1.23205078f,  0.7320508f};
    const float sigma{0.640000045f};
    const float lambda{1.14285719f};
    const float phy{0.0f};
    const float gamma{0.5f};
    const size_t filter_dimension_seed{2};

    const std::vector<float> expected{
        -2.81440061e-05f, -0.00234148954f, 0.0324483067f, 0.0615648068f,   -0.013143762f,
        0.0008446796f,    0.01288948f,     -0.210444331f, -0.07977704f,    0.0185685251f,
        -0.007318391f,    0.00703506032f,  0.388561845f,  0.00703506032f,  -0.007318391f,
        0.0185685251f,    -0.07977704f,    -0.210444331f, 0.01288948f,     0.0008446796,
        -0.013143762f,    0.0615648068f,   0.0324483067f, -0.00234148954f, -2.81440061e-05f};

    std::vector<float> results(5 * 5);
    const struct alus::featurextractiongabor::FilterBankItemParameters parameters {
        theta_x, theta_y, sigma, lambda, phy, gamma, filter_dimension_seed
    };
    alus::featurextractiongabor::ComputeFilterBankItem(results.data(), 25, parameters);
    ASSERT_THAT(results, Pointwise(FloatNear(1e-7), expected));
}

TEST(FilterBank, CreatesGaborFilterBankCorrectly) {
    const size_t orientations{6};
    const size_t frequencies{4};
    const auto& bank = alus::featurextractiongabor::CreateGaborFilterBank(orientations, frequencies);

    ASSERT_THAT(bank, SizeIs(orientations * frequencies));

    const float* etalon_data = reinterpret_cast<float*>(_binary_filter_banks_gabor_6or_4f_bin_start);
    const size_t etalon_size = _binary_filter_banks_gabor_6or_4f_bin_end - _binary_filter_banks_gabor_6or_4f_bin_start;
    size_t elements_compared{};
    double etalon_sum{};
    double results_sum{};
    size_t etalon_index{};
    for (auto&& b : bank) {
        for (auto&& f : b.filter_buffer) {
            etalon_sum += etalon_data[etalon_index];
            results_sum += f;
            EXPECT_THAT(f, FloatNear(etalon_data[etalon_index++], 1e-8));
            elements_compared++;
        }
    }

    ASSERT_THAT(etalon_index * sizeof(float), Eq(etalon_size));
    ASSERT_THAT(elements_compared, Eq(5304));
    ASSERT_THAT(etalon_sum, DoubleNear(results_sum, 1e-7));
}

}  // namespace