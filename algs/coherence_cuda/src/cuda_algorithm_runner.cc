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
#include "cuda_algorithm_runner.h"

#include <array>
#include <mutex>
#include <thread>
#include <vector>

namespace alus {
namespace coherence_cuda {
struct CUDAAlgorithmRunner::ThreadParams
{
    std::vector<CohTile> tiles;
    std::mutex tiles_mutex;
    size_t read_tile_count = 0;
    std::mutex write_mutex;
    std::mutex exception_mutex;
    std::exception_ptr exception;
};


void CUDAAlgorithmRunner::ThreadRun(CUDAAlgorithmRunner* algo, ThreadParams* params){
    try {
        std::array<std::vector<float>, 4> data_in;
        std::vector<float> data_out;
        while (true) {
            {
                std::unique_lock l(params->exception_mutex);
                if(params->exception != nullptr) return;
            }
            CohTile tile = {};
            {
                std::unique_lock l(params->tiles_mutex);
                if (params->read_tile_count == params->tiles.size()) return;

                tile = params->tiles[params->read_tile_count];
                params->read_tile_count++;
            }
            const auto& tile_in = tile.GetTileIn();
            const auto& tile_out = tile.GetTileOut();
            data_out.resize(tile_out.GetXSize() * tile_out.GetYSize());
            const size_t tile_sz = tile_in.GetXSize() * tile_in.GetYSize();
            if(tile_sz > data_in.at(0).size())
            {
                for(auto& vec : data_in) {
                    vec.resize(tile_sz);
                }
            }

            for(int band_nr = 1 ; band_nr <= 4; band_nr++)
            {
                algo->tile_reader_->ReadTile(tile_in, data_in.at(band_nr - 1).data(), band_nr);
            }
            algo->algo_->TileCalc(tile, data_in, data_out);
            {
                std::unique_lock l(params->write_mutex);
                algo->tile_writer_->WriteTile(tile_out, data_out.data(), data_out.size());
            }
        }
    }
    catch(std::exception& e) {
        std::unique_lock l(params->exception_mutex);
        if(params->exception == nullptr){
            params->exception = std::current_exception();
        }
    }
}

void CUDAAlgorithmRunner::Run() {
    auto tiles = tile_provider_->GetTiles();
    ThreadParams params = {};
    params.tiles = tile_provider_->GetTiles();
    algo_->PreTileCalc();

    std::vector<std::thread> thread_vec;
    constexpr size_t N_EXTRA_THREADS = 4;
    for(size_t i = 0; i < N_EXTRA_THREADS; i++){
        thread_vec.emplace_back(ThreadRun, this, &params);
    }

    for(auto& t : thread_vec){
        t.join();
    }

    if(params.exception != nullptr){
        std::rethrow_exception(params.exception);
    }
}

CUDAAlgorithmRunner::CUDAAlgorithmRunner(GdalTileReader* tile_reader, IDataTileWriter* tile_writer,
                                         ITileProvider* tile_provider, IAlgoCuda* algorithm)
    : tile_reader_{tile_reader}, tile_writer_{tile_writer}, tile_provider_{tile_provider}, algo_{algorithm} {}

}  // namespace coherence_cuda
}  // namespace alus