/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.io.orbits.sentinel1.SentinelPODOrbitFile.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/s1tbx). It was originally stated:
 *
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
#include "io/orbits/sentinel1/sentinel_p_o_d_orbit_file.h"

#include <algorithm>
#include <fstream>

#include <boost/algorithm/string/replace.hpp>

#include "snap-engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-engine-utilities/util/maths.h"
#include "snap-core/util/alus_utils.h"
#include "snap-core/util/string_utils.h"
#include "snap-core/util/system_utils.h"

namespace alus {
namespace s1tbx {

std::vector<std::shared_ptr<snapengine::OrbitVector>> SentinelPODOrbitFile::GetOrbitData(double start_utc,
                                                                                         double end_utc) {
    auto start_osv = std::make_shared<snapengine::OrbitVector>(start_utc);
    int start_idx = static_cast<int>(
        std::binary_search(osv_list_.begin(), osv_list_.end(), start_osv, snapengine::OrbitVector::Compare));

    if (start_idx < 0) {
        size_t insertion_pt = -(start_idx + 1);
        if (insertion_pt == osv_list_.size()) {
            start_idx = insertion_pt - 1;
        } else if (insertion_pt <= 0) {
            start_idx = 0;
        } else {
            start_idx = insertion_pt - 1;
        }
    }

    auto end_osv = std::make_shared<snapengine::OrbitVector>(end_utc);
    int end_idx = static_cast<int>(
        std::binary_search(osv_list_.begin(), osv_list_.end(), end_osv, snapengine::OrbitVector::Compare));

    if (end_idx < 0) {
        size_t insertion_pt = -(end_idx + 1);
        if (insertion_pt == osv_list_.size()) {
            end_idx = insertion_pt - 1;
        } else if (insertion_pt == 0) {
            end_idx = 0;
        } else {
            end_idx = insertion_pt;
        }
    }

    start_idx -= 3;
    end_idx += 3;

    int num_osv = end_idx - start_idx + 1;
    std::vector<std::shared_ptr<snapengine::OrbitVector>> orbit_data_list(num_osv);
    int idx = start_idx;
    for (int i = 0; i < num_osv; i++) {
        orbit_data_list.at(i) = osv_list_.at(idx);
        idx++;
    }

    return orbit_data_list;
}
std::shared_ptr<snapengine::OrbitVector> SentinelPODOrbitFile::GetOrbitData(double utc) {
    int num_vectors = osv_list_.size();
    double t0 = osv_list_.at(0)->utc_mjd_;
    double t_n = osv_list_.at(num_vectors - 1)->utc_mjd_;

    int num_vec_poly_fit = poly_degree_ + 1;  // 4;
    int half_num_vec_poly_fit = num_vec_poly_fit / 2;
    std::vector<int> vector_indices(num_vec_poly_fit);

    int vec_idx = static_cast<int>((utc - t0) / (t_n - t0) * (num_vectors - 1));
    if (vec_idx <= half_num_vec_poly_fit - 1) {
        for (int i = 0; i < num_vec_poly_fit; i++) {
            vector_indices.at(i) = i;
        }
    } else if (vec_idx >= num_vectors - half_num_vec_poly_fit) {
        for (int i = 0; i < num_vec_poly_fit; i++) {
            vector_indices.at(i) = num_vectors - num_vec_poly_fit + i;
        }
    } else {
        for (int i = 0; i < num_vec_poly_fit; i++) {
            vector_indices.at(i) = vec_idx - half_num_vec_poly_fit + 1 + i;
        }
    }

    std::vector<double> time_array(num_vec_poly_fit);
    std::vector<double> x_pos_array(num_vec_poly_fit);
    std::vector<double> y_pos_array(num_vec_poly_fit);
    std::vector<double> z_pos_array(num_vec_poly_fit);
    std::vector<double> x_vel_array(num_vec_poly_fit);
    std::vector<double> y_vel_array(num_vec_poly_fit);
    std::vector<double> z_vel_array(num_vec_poly_fit);

    for (int i = 0; i < num_vec_poly_fit; i++) {
        time_array.at(i) = osv_list_.at(vector_indices.at(i))->utc_mjd_ - t0;
        x_pos_array.at(i) = osv_list_.at(vector_indices.at(i))->x_pos_;
        y_pos_array.at(i) = osv_list_.at(vector_indices.at(i))->y_pos_;
        z_pos_array.at(i) = osv_list_.at(vector_indices.at(i))->z_pos_;
        x_vel_array.at(i) = osv_list_.at(vector_indices.at(i))->x_vel_;
        y_vel_array.at(i) = osv_list_.at(vector_indices.at(i))->y_vel_;
        z_vel_array.at(i) = osv_list_.at(vector_indices.at(i))->z_vel_;
    }

    auto a = snapengine::Maths::CreateVandermondeMatrix(time_array, poly_degree_);
    std::vector<double> x_pos_coeff = snapengine::Maths::PolyFit(a, x_pos_array);
    std::vector<double> y_pos_coeff = snapengine::Maths::PolyFit(a, y_pos_array);
    std::vector<double> z_pos_coeff = snapengine::Maths::PolyFit(a, z_pos_array);
    std::vector<double> x_vel_coeff = snapengine::Maths::PolyFit(a, x_vel_array);
    std::vector<double> y_vel_coeff = snapengine::Maths::PolyFit(a, y_vel_array);
    std::vector<double> z_vel_coeff = snapengine::Maths::PolyFit(a, z_vel_array);

    double normalized_time = utc - t0;

    return std::make_shared<snapengine::OrbitVector>(utc, snapengine::Maths::PolyVal(normalized_time, x_pos_coeff),
                                                     snapengine::Maths::PolyVal(normalized_time, y_pos_coeff),
                                                     snapengine::Maths::PolyVal(normalized_time, z_pos_coeff),
                                                     snapengine::Maths::PolyVal(normalized_time, x_vel_coeff),
                                                     snapengine::Maths::PolyVal(normalized_time, y_vel_coeff),
                                                     snapengine::Maths::PolyVal(normalized_time, z_vel_coeff));
}

boost::filesystem::path SentinelPODOrbitFile::RetrieveOrbitFile(std::string_view orbit_type) {
    double state_vector_time = abs_root_->GetAttributeUtc(snapengine::AbstractMetadata::STATE_VECTOR_TIME)->GetMjd();
    auto calendar = abs_root_->GetAttributeUtc(snapengine::AbstractMetadata::STATE_VECTOR_TIME)->GetAsCalendar();
    //    int year = calendar.get(Calendar.YEAR);
    int year = calendar.date().year();
    //    int month = calendar.get(Calendar.MONTH) + 1;  // zero based
    int month = calendar.date().month().as_enum();
    //    int day = calendar.get(Calendar.DAY_OF_MONTH);
    //    todo: remove [[maybe_unused]] below when implementing further
    [[maybe_unused]] int day = calendar.date().day();
    //    int hour = calendar.get(Calendar.HOUR_OF_DAY);
    [[maybe_unused]] int hour = calendar.time_of_day().hours();
    //    int minute = calendar.get(Calendar.MINUTE);
    [[maybe_unused]] int minute = calendar.time_of_day().minutes();
    //    int second = calendar.get(Calendar.SECOND);
    [[maybe_unused]] int second = calendar.time_of_day().seconds();

    std::string mission_prefix = GetMissionPrefix(abs_root_);
    orbit_file_ = FindOrbitFile(mission_prefix, orbit_type, state_vector_time, year, month);
    //    todo:support download using asio
    //    if (orbit_file_) {
    //        orbit_file_ = DownloadFromQCRestAPI(
    //            mission_prefix, orbit_type, year, month, day, hour, minute, second, state_vector_time);
    //    }
    //    if (orbit_file_) {
    //        orbit_file_ = DownloadFromStepAuxdata(mission_prefix, orbit_type, year, month, day, state_vector_time);
    //    }

    if (!orbit_file_) {
        std::string time_str = abs_root_->GetAttributeUtc(snapengine::AbstractMetadata::STATE_VECTOR_TIME)->Format();
        // todo:replace boost directory
        boost::filesystem::path dest_folder = GetDestFolder(mission_prefix, orbit_type, year, month);
        throw std::runtime_error("No valid orbit file found for " + time_str +
                                 "\nOrbit files may be downloaded from https://qc.sentinel1.eo.esa.int/" +
                                 "\nand placed in " + boost::filesystem::canonical(dest_folder).string());
    }

    if (!boost::filesystem::exists(*orbit_file_)) {
        throw std::runtime_error("SentinelPODOrbitFile: Unable to find POD orbit file");
    }

    // read content of the orbit file
    ReadOrbitFile();

    return *orbit_file_;
}

std::string SentinelPODOrbitFile::GetMissionPrefix(std::shared_ptr<snapengine::MetadataElement> abs_root) {
    std::string mission = abs_root->GetAttributeString(snapengine::AbstractMetadata::MISSION);
    return "S1" + mission.substr(mission.length() - 1);
}

boost::filesystem::path SentinelPODOrbitFile::GetDestFolder(std::string_view mission_prefix,
                                                            std::string_view orbit_type, int year, int month) {
    std::string pref_orbit_path;
    // just using current directory (need to discuss where to keep this)
    //    std::string pref_orbit_path = ".";
    // todo: this should return correct data directory: ~/.snap/auxdata/Orbits/Sentinel-1/POEORB
    // todo: probably enough to use custom dedicated directory for start
    if (orbit_type.rfind(RESTITUTED, 0) == 0) {
        //        std::string def =
        //            SystemUtils.getAuxDataPath().resolve("Orbits").resolve("Sentinel-1").resolve("RESORB").toString();
        //        pref_orbit_path = Settings.instance().get("OrbitFiles.sentinel1RESOrbitPath", def);
    } else {
        //        std::string def =
        //            SystemUtils.getAuxDataPath().resolve("Orbits").resolve("Sentinel-1").resolve("POEORB").toString();
        //        pref_orbit_path = Settings.instance().get("OrbitFiles.sentinel1POEOrbitPath", def);
    }
    boost::filesystem::path dest_folder(/*pref_orbit_path + boost::filesystem::path::preferred_separator +*/
                                        // todo:hardcoded directory probably want to provide this from command line,
                                        // need to talk this over with others
                                        snapengine::SystemUtils::GetAuxDataPath().string() +  std::string(mission_prefix) +
                                        boost::filesystem::path::preferred_separator + std::to_string(year) +
                                        boost::filesystem::path::preferred_separator +
                                        snapengine::StringUtils::PadNum(month, 2, '0'));

    if (month < 10) {
        boost::filesystem::path old_folder(/*pref_orbit_path + boost::filesystem::path::preferred_separator +*/
                                           std::string(mission_prefix) + boost::filesystem::path::preferred_separator +
                                           std::to_string(year) + boost::filesystem::path::preferred_separator +
                                           std::to_string(month));
        if (boost::filesystem::exists(old_folder) && boost::filesystem::is_directory(old_folder)) {
            // rename
            rename(old_folder, dest_folder);
        }
    }
    boost::filesystem::create_directories(dest_folder);
    return dest_folder;
}

std::optional<boost::filesystem::path> SentinelPODOrbitFile::FindOrbitFile(std::string_view mission_prefix,
                                                                           std::string_view orbit_type,
                                                                           double state_vector_time, int year,
                                                                           int month) {

    if (alus::snapengine::AlusUtils::IsOrbitFileAssigned()) {
        const auto orbit_file_path = alus::snapengine::AlusUtils::GetOrbitFilePath();
        if (IsWithinRange(orbit_file_path.filename().string(), state_vector_time)) {
            return orbit_file_path;
        }
        else {
            std::cerr << "Orbit file '" << orbit_file_path
                      << "' is not correct for given input (start and end time out of range)." << std::endl;
        }
    } else {
        boost::filesystem::path orbit_file_folder = GetDestFolder(mission_prefix, orbit_type, year, month);

        if (!(boost::filesystem::exists(orbit_file_folder) && boost::filesystem::is_directory(orbit_file_folder))) {
            return std::nullopt;
        }
        //    std::vector<boost::filesystem::path> files = orbit_file_folder.listFiles(new S1OrbitFileFilter(prefix));
        for (boost::filesystem::directory_entry& file : boost::filesystem::directory_iterator(orbit_file_folder)) {
            if (IsWithinRange(file.path().filename().string(), state_vector_time)) {
                return file;
            }
        }
    }

    return std::nullopt;
}

bool SentinelPODOrbitFile::IsWithinRange(std::string_view filename, double state_vector_time) {
    try {
        std::shared_ptr<snapengine::Utc> utc_end = SentinelPODOrbitFile::GetValidityStopFromFilenameUTC(filename);
        std::shared_ptr<snapengine::Utc> utc_start = SentinelPODOrbitFile::GetValidityStartFromFilenameUTC(filename);
        if (utc_start != nullptr && utc_end != nullptr) {
            return state_vector_time >= utc_start->GetMjd() && state_vector_time < utc_end->GetMjd();
        }
        return false;
    } catch (std::exception& e) {
        return false;
    }
}

std::shared_ptr<snapengine::Utc> SentinelPODOrbitFile::GetValidityStartFromFilenameUTC(std::string_view filename) {
    if (filename.substr(41, 1) == "V") {
        std::string val = ExtractTimeFromFilename(filename, 42);
        return snapengine::Utc::Parse(val, DATE_FORMAT);
    }
    return nullptr;
}

std::shared_ptr<snapengine::Utc> SentinelPODOrbitFile::GetValidityStopFromFilenameUTC(std::string_view filename) {
    if (filename.substr(41, 1) == "V") {
        std::string val = ExtractTimeFromFilename(filename, 58);
        return snapengine::Utc::Parse(val, DATE_FORMAT);
    }
    return nullptr;
}

std::string SentinelPODOrbitFile::ExtractTimeFromFilename(std::string_view filename, int offset) {
    std::string str = std::string(filename.substr(offset, 15));
    boost::replace_all(str, "T", "-");
    return str;
}

void SentinelPODOrbitFile::ReadOrbitFile() {
    //    todo: find a similar solution for caching
    //    std::vector<snapengine::OrbitVector> cached_o_s_v_list = GetCache()->Get(orbit_file_);

    //    if (!cached_o_s_v_list.empty()) {
    //        osv_list_ = cached_o_s_v_list;
    //        return;
    //    }

    //    todo: add zip file support!
    /*
    final DocumentBuilderFactory documentFactory = DocumentBuilderFactory.newInstance();
    final DocumentBuilder documentBuilder = documentFactory.newDocumentBuilder();

    final Document doc;
    if (orbitFile.getName().toLowerCase().endsWith(".zip")) {
        final ZipFile productZip = new ZipFile(orbitFile, ZipFile.OPEN_READ);
        final Enumeration < ? extends ZipEntry > entries = productZip.entries();
        final ZipEntry zipEntry = entries.nextElement();

        doc = documentBuilder.parse(productZip.getInputStream(zipEntry));
    } else {
        doc = documentBuilder.parse(orbitFile);
    }*/
    pugi::xml_document doc;
    doc.load_file(boost::filesystem::canonical(*orbit_file_, boost::filesystem::current_path()).c_str());
    pugi::xpath_variable_set vars;
    vars.add("name", pugi::xpath_type_string);
    std::string query_str{"//"};
    query_str.append("Earth_Explorer_File");
    pugi::xpath_query query(query_str.data(), &vars);
    pugi::xpath_node_set query_result = query.evaluate_node_set(doc);
    // todo:  looks like some normalization is default in pugixml not sure if this is the same, ignore until relevant
    //    doc.getDocumentElement().normalize();
    if (query_result.size() != 1) {
        throw std::runtime_error("SentinelPODOrbitFile.readOrbitFile: ERROR found too many Earth_Explorer_File " +
                                 std::to_string(query_result.size()));
    }

    //    todo:check what value these get here, hoping for null node (iterator break probably expects null node)
    pugi::xml_node fixed_header_node;
    pugi::xml_node variable_header_node;
    pugi::xml_node list_of_o_s_v_s_node;
    for (auto& file_child_node : query_result.first().node()) {
        if (file_child_node.name() == std::string("Earth_Explorer_Header")) {
            for (auto& header_child_node : file_child_node) {
                if (header_child_node.name() == std::string("Fixed_Header")) {
                    fixed_header_node = header_child_node;
                } else if (header_child_node.name() == std::string("Variable_Header")) {
                    variable_header_node = header_child_node;
                }
            }
        } else if (file_child_node.name() == std::string("Data_Block")) {
            for (auto& data_block_child_node : file_child_node) {
                if (data_block_child_node.name() == std::string("List_of_OSVs")) {
                    list_of_o_s_v_s_node = data_block_child_node;
                }
            }
        }
        // break iterations
        //        todo:check if this works like expected (null node)
        if (fixed_header_node && variable_header_node && list_of_o_s_v_s_node) {
            break;
        }
    }
    if (fixed_header_node) {
        ReadFixedHeader(fixed_header_node);
    }

    // Don't need anything from Variable_Header.

    if (list_of_o_s_v_s_node) {
        osv_list_ = ReadOSVList(list_of_o_s_v_s_node);
    }
    CheckOrbitFileValidity();
    //    getCache().put(orbitFile, osvList);
}

void SentinelPODOrbitFile::CheckOrbitFileValidity() {
    double state_vector_time = abs_root_->GetAttributeUtc(snapengine::AbstractMetadata::STATE_VECTOR_TIME)->GetMjd();
    std::string validity_start_time_str = GetValidityStartFromHeader();
    std::string validity_stop_time_str = GetValidityStopFromHeader();
    double validity_start_time_m_j_d = SentinelPODOrbitFile::ToUtc(validity_start_time_str)->GetMjd();
    double validity_stop_time_m_j_d = SentinelPODOrbitFile::ToUtc(validity_stop_time_str)->GetMjd();
    if (state_vector_time < validity_start_time_m_j_d || state_vector_time > validity_stop_time_m_j_d) {
        throw std::runtime_error("Product acquisition time is not within the validity period of the orbit");
    }
}
std::shared_ptr<snapengine::Utc> SentinelPODOrbitFile::ToUtc(std::string_view str) {
    return snapengine::Utc::Parse(ConvertUtc(str), ORBIT_DATE_FORMAT);
}
std::string SentinelPODOrbitFile::ConvertUtc(std::string_view utc) {
    std::string str = std::string(utc);
    boost::replace_all(str, "UTC=", "");
    boost::replace_all(str, "T", " ");
    return str;
}

std::string SentinelPODOrbitFile::GetValidityStartFromHeader() {
    if (fixed_header_ != nullptr) {
        return fixed_header_->validity_start_;
    }
    return "";
}

std::string SentinelPODOrbitFile::GetValidityStopFromHeader() {
    if (fixed_header_ != nullptr) {
        return fixed_header_->validity_stop_;
    }
    return "";
}
void SentinelPODOrbitFile::ReadFixedHeader(pugi::xml_node& fixed_header_node) {
    std::string mission;
    std::string file_type;
    std::string validity_start;
    std::string validity_stop;

    for (auto& fixed_header_child_node : fixed_header_node) {
        if (fixed_header_child_node.name() == std::string("Mission")) {
            mission = fixed_header_child_node.text().as_string();
        } else if (fixed_header_child_node.name() == std::string("File_Type")) {
            file_type = fixed_header_child_node.text().as_string();
        } else if (fixed_header_child_node.name() == std::string("Validity_Period")) {
            for (auto& validity_period_child_node : fixed_header_child_node) {
                if (validity_period_child_node.name() == std::string("Validity_Start")) {
                    validity_start = validity_period_child_node.text().as_string();
                } else if (validity_period_child_node.name() == std::string("Validity_Stop")) {
                    validity_stop = validity_period_child_node.text().as_string();
                }
            }
        }

        if (!mission.empty() && !file_type.empty() && !validity_start.empty() && !validity_stop.empty()) {
            fixed_header_ = std::make_unique<FixedHeader>(mission, file_type, validity_start, validity_stop);
            break;
        }
    }
}
struct PredAttrCount {
    bool operator()(pugi::xml_attribute attr) const { return strcmp(attr.name(), "count") == 0; }
};
std::vector<std::shared_ptr<snapengine::OrbitVector>> SentinelPODOrbitFile::ReadOSVList(
    pugi::xml_node& list_of_o_s_v_s_node) {
    int count = std::stoi(list_of_o_s_v_s_node.find_attribute(PredAttrCount()).value());
    std::vector<std::shared_ptr<snapengine::OrbitVector>> osv_list;
    int osv_cnt = 0;
    auto child_node = list_of_o_s_v_s_node.first_child();

    while (child_node) {
        if (child_node.name() == std::string("OSV")) {
            osv_cnt++;
            osv_list.push_back(ReadOneOSV(child_node));
        }
        child_node = child_node.next_sibling();
    }

    std::sort(osv_list.begin(), osv_list.end(), snapengine::OrbitVector::Compare);

    if (count != osv_cnt) {
        std::cerr << "SentinelPODOrbitFile::ReadOSVList: WARNING List_of_OSVs count = " << count << " but found only "
                  << osv_cnt << " OSV" << std::endl;
    }
    return osv_list;
}

std::shared_ptr<snapengine::OrbitVector> SentinelPODOrbitFile::ReadOneOSV(const pugi::xml_node& osv_node) {
    std::string utc;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;
    auto child_node = osv_node.first_child();
    while (child_node) {
        if (child_node.name() == std::string("UTC")) {
            utc = child_node.text().as_string();
        } else if (child_node.name() == std::string("X")) {
            x = std::stod(child_node.text().as_string());
        } else if (child_node.name() == std::string("Y")) {
            y = std::stod(child_node.text().as_string());
        } else if (child_node.name() == std::string("Z")) {
            z = std::stod(child_node.text().as_string());
        } else if (child_node.name() == std::string("VX")) {
            vx = std::stod(child_node.text().as_string());
        } else if (child_node.name() == std::string("VY")) {
            vy = std::stod(child_node.text().as_string());
        } else if (child_node.name() == std::string("VZ")) {
            vz = std::stod(child_node.text().as_string());
        }
        child_node = child_node.next_sibling();
    }
    double utc_time = ToUtc(utc)->GetMjd();
    return std::make_shared<snapengine::OrbitVector>(utc_time, x, y, z, vx, vy, vz);
}
std::optional<boost::filesystem::path> SentinelPODOrbitFile::DownloadFromQCRestAPI(std::string_view mission_prefix,
                                                                                   std::string_view orbit_type,
                                                                                   int year, int month, int day,
                                                                                   int hour, int minute, int second,
                                                                                   double state_vector_time) {
    std::string orb_product_type = orbit_type == RESTITUTED ? "AUX_RESORB" : "AUX_POEORB";
    std::string date = std::to_string(year) + "-" + std::to_string(month) + "-" + std::to_string(day);

    std::string end_point = "https://qc.sentinel1.eo.esa.int/api/v1/?product_type=" + orb_product_type;
    end_point += "&validity_stop__gt=" + date + "T23:59:59";
    end_point += "&validity_start__lt=" + date + "T" + std::to_string(hour) + ":" + std::to_string(minute) + ":" +
                 std::to_string(second);
    end_point += "&ordering=-creation_date&page_size=1";

    //    try(CloseableHttpClient httpClient = HttpClients.createDefault()) {
    //            HttpGet httpGet = new HttpGet(endPoint);
    //
    //            final StringBuilder outputBuilder = new StringBuilder();
    //            try
    //                (final CloseableHttpResponse response = httpClient.execute(httpGet)) {
    //                    final int status = response.getStatusLine().getStatusCode();
    //
    //                    try
    //                        (final BufferedReader rd =
    //                             new BufferedReader(new InputStreamReader(response.getEntity().getContent()))) {
    //                            String line;
    //                            while ((line = rd.readLine()) != null) {
    //                                outputBuilder.append(line);
    //                            }
    //                        }
    //                    response.close();
    //
    //                    if (status == 200) {
    //                        final JSONParser parser = new JSONParser();
    //                        JSONObject json = (JSONObject)parser.parse(outputBuilder.toString());
    //                        if (json.containsKey("results")) {
    //                            JSONArray results = (JSONArray)json.get("results");
    //                            if (!results.isEmpty()) {
    //                                JSONObject result = (JSONObject)results.get(0);
    //                                if (result.containsKey("remote_url")) {
    //                                    String remoteURL = (String)result.get("remote_url");
    //
    //                                    getQCFile(missionPrefix, orbitType, year, month, remoteURL);
    //                                }
    //                            }
    //                        }
    //                    }
    //                }
    //            catch (IOException e) {
    //                System.out.println("Exception calling QC Rest API:  " + e.getMessage());
    //                throw e;
    //            }
    //        }

    std::optional<boost::filesystem::path> orbit_file =
        FindOrbitFile(mission_prefix, orbit_type, state_vector_time, year, month);
    return orbit_file;
}

}  // namespace s1tbx
}  // namespace alus