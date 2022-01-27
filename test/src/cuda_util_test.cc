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

#include "cuda_mem_arena.h"

namespace {

template <class T>
bool VerifyAlignment(T* p) {
    auto int_ptr = reinterpret_cast<uintptr_t>(p);
    return int_ptr != 0 && (int_ptr % alignof(T)) == 0;
}

}  // namespace

TEST(CudaUtilTest, MemArenaPointersAlignedCorrectly) {
    alus::cuda::MemArena arena;
    arena.ReserveMemory(10000);  // NOLINT

    for (int i = 0; i < 10; i++) {  // NOLINT
        auto* p_i = arena.AllocArray<int32_t>(1);
        auto* p_s = arena.AllocArray<int16_t>(1);
        auto* p_d = arena.AllocArray<double>(1);
        auto* p_c = arena.AllocArray<uint8_t>(1);
        ASSERT_TRUE(VerifyAlignment(p_i));
        ASSERT_TRUE(VerifyAlignment(p_s));
        ASSERT_TRUE(VerifyAlignment(p_d));
        ASSERT_TRUE(VerifyAlignment(p_c));
    }
}

TEST(CudaUtilTest, MemArenaAllocationWorksCorrectly) {
    alus::cuda::MemArena arena;
    arena.ReserveMemory(10000);  // NOLINT

    auto* p1 = arena.AllocArray<double>(500);  // NOLINT
    auto* p2 = arena.AllocArray<double>(500);  // NOLINT
    VerifyAlignment(p1);
    VerifyAlignment(p2);

    EXPECT_ANY_THROW(arena.AllocArray<double>(500));

    arena.Reset();

    auto* p3 = arena.AllocArray<double>(500);  // NOLINT
    auto* p4 = arena.AllocArray<double>(500);  // NOLINT
    VerifyAlignment(p3);
    VerifyAlignment(p4);
}
