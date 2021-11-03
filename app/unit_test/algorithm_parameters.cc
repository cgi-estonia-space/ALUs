#include "algorithm_parameters.h"

#include <experimental/filesystem>
#include <fstream>
#include <stdexcept>

#include "gmock/gmock.h"

namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

using namespace alus::app;
using namespace std::experimental;

TEST(AlgorithmParametersTest, throwsOnInvalidSyntax) {
    EXPECT_THROW(AlgorithmParameters(""), std::runtime_error);
    EXPECT_THROW(AlgorithmParameters("coherence"), std::runtime_error);
    EXPECT_THROW(AlgorithmParameters("coherence:"), std::runtime_error);
    EXPECT_THROW(AlgorithmParameters("coherence:param"), std::runtime_error);
    EXPECT_THROW(AlgorithmParameters("coherence:param="), std::runtime_error);
    EXPECT_THROW(AlgorithmParameters("coherence:param=value,"), std::runtime_error);
    EXPECT_THROW(AlgorithmParameters("coherence:param=value,param=value"), std::runtime_error);
    EXPECT_THROW(AlgorithmParameters("coherence:param=value,param1=value,"), std::runtime_error);
}

TEST(AlgorithmParametersTest, parsesCorrectlySingleEntityParameters) {
    AlgorithmParameters test("testalgo:param=value,param2=2,param3=value,tere=mistoimps,weird=val ue");
    ASSERT_THAT(test.GetAlgorithmName(), Eq("testalgo"));
    ASSERT_THAT(test.GetParameters(),
                UnorderedElementsAre(Pair("param", "value"),
                                     Pair("param2", "2"),
                                     Pair("param3", "value"),
                                     Pair("tere", "mistoimps"),
                                     Pair("weird", "val ue")));
}

TEST(AlgorithmParametersTest, parsesCorrectlyParametersWithoutAlgorithmName) {
    AlgorithmParameters test("first=1,second=2,param3=5456.0932,par am=1e-6,last_one=OK12");
    ASSERT_THAT(test.GetAlgorithmName(), IsEmpty());
    ASSERT_THAT(test.GetParameters(),
                UnorderedElementsAre(Pair("first", "1"),
                                     Pair("second", "2"),
                                     Pair("param3", "5456.0932"),
                                     Pair("par am", "1e-6"),
                                     Pair("last_one", "OK12")));
}

TEST(AlgorithmParametersTest, parsesCorrectlyParametersWithoutAlgorithmNameWithWkt) {
    AlgorithmParameters test(
        "first=yes,second=2,param3=5.930,wkt=POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10)),par am=1e-6");
    ASSERT_THAT(test.GetAlgorithmName(), IsEmpty());
    ASSERT_THAT(test.GetParameters(),
                UnorderedElementsAre(Pair("first", "yes"),
                                     Pair("second", "2"),
                                     Pair("param3", "5.930"),
                                     Pair("par am", "1e-6"),
                                     Pair("wkt", "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")));
}

TEST(AlgorithmParametersTest, parsesCorrectlyMulipleAlgorithmParameters) {
    const auto& results = AlgorithmParameters::TryCreateFrom(
        "terrain-correction:useAvgSceneHeight=true,points=10;coherence:subtract_flat_earth_phase=false,rg=65.90,az=140;"
        "backgeocoding:interpolation=bilinear");

    ASSERT_THAT(results, SizeIs(3));
    EXPECT_THAT(results.at("terrain-correction"),
                UnorderedElementsAre(Pair("useAvgSceneHeight", "true"), Pair("points", "10")));
    EXPECT_THAT(
        results.at("coherence"),
        UnorderedElementsAre(Pair("subtract_flat_earth_phase", "false"), Pair("rg", "65.90"), Pair("az", "140")));
    EXPECT_THAT(results.at("backgeocoding"), UnorderedElementsAre(Pair("interpolation", "bilinear")));
}

TEST(AlgorithmParametersTest, parsesCorrectlyMultipleAlgorithmParametersFromFile) {
    const std::string parameter_file_name{"param_file_unit_test.txt"};
    const auto temp_file = filesystem::temp_directory_path() / parameter_file_name;
    filesystem::remove(temp_file);
    const auto temp_file_path{temp_file.string()};

    std::ofstream param_file_contents;
    param_file_contents.exceptions(std::ifstream::badbit);
    param_file_contents.open(temp_file_path, std::ios_base::out);
    param_file_contents << "terrain-correction:"
                        << "useAvgSceneHeight=true,"
                        << "points=10;"
                        << "coherence:subtract_flat_earth_phase=false,rg=65.90,az=140;"
                        << "backgeocoding:"
                        << "interpolation=bilinear";
    param_file_contents.close();

    const auto& results = AlgorithmParameters::TryCreateFromFile(temp_file_path);

    ASSERT_THAT(results, SizeIs(3));
    EXPECT_THAT(results.at("terrain-correction"),
                UnorderedElementsAre(Pair("useAvgSceneHeight", "true"), Pair("points", "10")));
    EXPECT_THAT(
        results.at("coherence"),
        UnorderedElementsAre(Pair("subtract_flat_earth_phase", "false"), Pair("rg", "65.90"), Pair("az", "140")));
    EXPECT_THAT(results.at("backgeocoding"), UnorderedElementsAre(Pair("interpolation", "bilinear")));
}

TEST(AlgorithmParametersTest, mergesDuplicateParametersWithWarnings) {
    const std::string parameter_file_name{"param_file_unit_test.txt"};
    const auto temp_file = filesystem::temp_directory_path() / parameter_file_name;
    filesystem::remove(temp_file);
    const auto temp_file_path{temp_file.string()};

    std::ofstream param_file_contents;
    param_file_contents.exceptions(std::ifstream::badbit);
    param_file_contents.open(temp_file_path, std::ios_base::out);
    param_file_contents << "terrain-correction:"
                        << "useAvgSceneHeight=true,"
                        << "points=10;"
                        << "TOPSAR-SPLIT:burst_start=0,burst_end=2;"
                        << "coherence:subtract_flat_earth_phase=false,rg=65.90,az=140;"
                        << "backgeocoding:"
                        << "interpolation=bilinear,altitude=yes";
    param_file_contents.close();

    const auto& file_params = AlgorithmParameters::TryCreateFromFile(temp_file_path);

    const auto& command_line_params = AlgorithmParameters::TryCreateFrom(
        "terrain-correction:useAvgSceneHeight=false,points=10;"
        "coherence:subtract_flat_earth_phase=false,rg=65.90,az=111;"
        "backgeocoding:interpolation=bilinear;scatter:window=65.9");

    std::string warnings{};
    const auto& results = AlgorithmParameters::MergeAndWarn(file_params, command_line_params, warnings);

    ASSERT_THAT(results, SizeIs(5));
    EXPECT_THAT(results.at("terrain-correction"),
                UnorderedElementsAre(Pair("useAvgSceneHeight", "false"), Pair("points", "10")));
    EXPECT_THAT(
        results.at("coherence"),
        UnorderedElementsAre(Pair("subtract_flat_earth_phase", "false"), Pair("rg", "65.90"), Pair("az", "111")));
    EXPECT_THAT(results.at("backgeocoding"),
                UnorderedElementsAre(Pair("interpolation", "bilinear"), Pair("altitude", "yes")));
    EXPECT_THAT(results.at("TOPSAR-SPLIT"), UnorderedElementsAre(Pair("burst_start", "0"), Pair("burst_end", "2")));
    EXPECT_THAT(results.at("scatter"), UnorderedElementsAre(Pair("window", "65.9")));

    EXPECT_THAT(warnings,
                HasSubstr("terrain-correction parameter 'useAvgSceneHeight' redeclared on command line, "
                          "overruling file configuration value to 'false'."));
    EXPECT_THAT(
        warnings,
        HasSubstr(
            "coherence parameter 'az' redeclared on command line, overruling file configuration value to '111'."));
    EXPECT_THAT(
        warnings,
        HasSubstr(
            "coherence parameter 'rg' redeclared on command line, overruling file configuration value to '65.90'."));
}

TEST(AlgorithmParametersTest, mergesParametersWithoutWarnings) {
    const std::string parameter_file_name{"param_file_unit_test.txt"};
    const auto temp_file = filesystem::temp_directory_path() / parameter_file_name;
    filesystem::remove(temp_file);
    const auto temp_file_path{temp_file.string()};

    std::ofstream param_file_contents;
    param_file_contents.exceptions(std::ifstream::badbit);
    param_file_contents.open(temp_file_path, std::ios_base::out);
    param_file_contents << "terrain-correction:"
                        << "useAvgSceneHeight=true,"
                        << "points=10;"
                        << "TOPSAR-SPLIT:burst_start=0,burst_end=2;"
                        << "coherence:subtract_flat_earth_phase=false,rg=65.90,az=140;"
                        << "backgeocoding:"
                        << "interpolation=bilinear,altitude=yes";
    param_file_contents.close();

    const auto& file_params = AlgorithmParameters::TryCreateFromFile(temp_file_path);

    const auto& command_line_params = AlgorithmParameters::TryCreateFrom(
        "terrain-correction:avgSceneHeight=10;"
        "coherence:orbit_degree=3;"
        "scatter:window=65.9");

    std::string warnings{};
    const auto& results = AlgorithmParameters::MergeAndWarn(file_params, command_line_params, warnings);

    ASSERT_THAT(results, SizeIs(5));
    EXPECT_THAT(results.at("terrain-correction"),
                UnorderedElementsAre(Pair("useAvgSceneHeight", "true"), Pair("points", "10"), Pair
                                     ("avgSceneHeight", "10")));
    EXPECT_THAT(
        results.at("coherence"),
        UnorderedElementsAre(Pair("subtract_flat_earth_phase", "false"), Pair("rg", "65.90"), Pair("az", "140"), Pair
            ("orbit_degree", "3")));
    EXPECT_THAT(results.at("backgeocoding"),
                UnorderedElementsAre(Pair("interpolation", "bilinear"), Pair("altitude", "yes")));
    EXPECT_THAT(results.at("TOPSAR-SPLIT"), UnorderedElementsAre(Pair("burst_start", "0"), Pair("burst_end", "2")));
    EXPECT_THAT(results.at("scatter"), UnorderedElementsAre(Pair("window", "65.9")));

    EXPECT_THAT(warnings, IsEmpty());
}

}  // namespace
