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

#include <mutex>
#include <thread>
#include <vector>

#include "alus_log.h"
#include "coherence_calc_cuda.h"
#include "cuda_copies.h"
#include "helper_cuda.h"
#include "tile_queue.h"

namespace alus {
namespace coherence_cuda {
struct CUDAAlgorithmRunner::ThreadParams {
    ThreadSafeTileQueue<CohTile> tiles;
    std::mutex write_mutex;
    std::mutex exception_mutex;
    std::exception_ptr exception;
    bool use_pinned_memory = false;
    size_t max_tile_sz = 0;
};

void CUDAAlgorithmRunner::ThreadRun(CUDAAlgorithmRunner* algo, ThreadParams* params) {
    coherence_cuda::ThreadContext ctx;
    try {
        CHECK_CUDA_ERRORS(cudaStreamCreate(&ctx.stream));
        ctx.h_buffer.Allocate(params->use_pinned_memory, params->max_tile_sz);
        CHECK_CUDA_ERRORS(cublasCreate(&ctx.handle));
        CHECK_CUDA_ERRORS(cublasSetStream(ctx.handle, ctx.stream));
        while (true) {
            {
                std::unique_lock l(params->exception_mutex);
                if (params->exception != nullptr) break;
            }
            CohTile tile = {};
            if (!params->tiles.PopFront(tile)) {
                break;
            }
            const auto& tile_in = tile.GetTileIn();
            const auto& tile_out = tile.GetTileOut();
            const size_t tile_in_sz = tile_in.GetXSize() * tile_in.GetYSize();

            // send tiles to GPU via the same host buffer, must synchronize stream when using it on host side
            // master I
            algo->tile_reader_->ReadTile(tile_in, ctx.h_buffer.Get(), 1);
            ctx.d_band_master_real.Resize(tile_in_sz);
            cuda::copyArrayAsyncH2D(ctx.d_band_master_real.Get(), ctx.h_buffer.Get(), tile_in_sz, ctx.stream);
            CHECK_CUDA_ERRORS(cudaStreamSynchronize(ctx.stream));
            // master Q
            algo->tile_reader_->ReadTile(tile_in, ctx.h_buffer.Get(), 2);
            ctx.d_band_master_imag.Resize(tile_in_sz);
            cuda::copyArrayAsyncH2D(ctx.d_band_master_imag.Get(), ctx.h_buffer.Get(), tile_in_sz, ctx.stream);
            CHECK_CUDA_ERRORS(cudaStreamSynchronize(ctx.stream));
            // slave I
            algo->tile_reader_->ReadTile(tile_in, ctx.h_buffer.Get(), 3);
            ctx.d_band_slave_real.Resize(tile_in_sz);
            cuda::copyArrayAsyncH2D(ctx.d_band_slave_real.Get(), ctx.h_buffer.Get(), tile_in_sz, ctx.stream);
            CHECK_CUDA_ERRORS(cudaStreamSynchronize(ctx.stream));
            // slave Q
            algo->tile_reader_->ReadTile(tile_in, ctx.h_buffer.Get(), 4);
            ctx.d_band_slave_imag.Resize(tile_in_sz);
            cuda::copyArrayAsyncH2D(ctx.d_band_slave_imag.Get(), ctx.h_buffer.Get(), tile_in_sz, ctx.stream);

            algo->algo_->TileCalc(tile, ctx);
            const size_t tile_out_sz = ctx.d_tile_out.GetElemCount();
            cuda::copyArrayAsyncD2H(ctx.h_buffer.Get(), ctx.d_tile_out.Get(), tile_out_sz, ctx.stream);
            CHECK_CUDA_ERRORS(cudaStreamSynchronize(ctx.stream));
            {
                std::unique_lock l(params->write_mutex);
                algo->tile_writer_->WriteTile(tile_out, ctx.h_buffer.Get(), tile_out_sz);
            }
        }
    } catch (std::exception& e) {
        std::unique_lock l(params->exception_mutex);
        if (params->exception == nullptr) {
            params->exception = std::current_exception();
        }
    }
    cublasDestroy(ctx.handle);
    cudaStreamDestroy(ctx.stream);
}

void CUDAAlgorithmRunner::Run() {
    ThreadParams params = {};
    auto tiles = tile_provider_->GetTiles();

    // find the biggest input tile, required for preallocating pinned memory
    int max_in_tile_sz = 0;
    for (const auto& e : tiles) {
        const auto& t = e.GetTileIn();
        const int tile_sz = t.GetYSize() * t.GetXSize();
        if (tile_sz > max_in_tile_sz) {
            max_in_tile_sz = tile_sz;
        }
    }

    // Coherence uses 4 input datasets -> 1 output dataset behind a mutex, this limit may change in the future
    constexpr size_t thread_limit = 4;
    const size_t n_threads = (tiles.size() / thread_limit) > 0 ? thread_limit : 1;

    // tradeoff between using pinned memory, no point using pinned memory if not doing enough transfers
    const bool use_pinned_memory = (tiles.size() / n_threads) >= 2;
    LOGD << "Coherence tiles = " << tiles.size() << " threads = " << n_threads
         << " transfer mode = " << (use_pinned_memory ? "pinned" : "paged");

    params.use_pinned_memory = use_pinned_memory;
    params.max_tile_sz = max_in_tile_sz;
    params.tiles.InsertData(std::move(tiles));

    algo_->PreTileCalc();

    std::vector<std::thread> thread_vec;
    for (size_t i = 0; i < n_threads; i++) {
        thread_vec.emplace_back(ThreadRun, this, &params);
    }

    // worker threads started, now wait for them to finish
    for (auto& t : thread_vec) {
        t.join();
    }

    if (params.exception != nullptr) {
        std::rethrow_exception(params.exception);
    }
}

CUDAAlgorithmRunner::CUDAAlgorithmRunner(GdalTileReader* tile_reader, IDataTileWriter* tile_writer,
                                         ITileProvider* tile_provider, CohCuda* algorithm)
    : tile_reader_{tile_reader}, tile_writer_{tile_writer}, tile_provider_{tile_provider}, algo_{algorithm} {}

}  // namespace coherence_cuda
}  // namespace alus