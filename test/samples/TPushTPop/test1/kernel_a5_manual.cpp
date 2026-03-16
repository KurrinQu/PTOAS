#include "pto/pto-inst.hpp"

using namespace pto;

static void matmul_tpush_tpop_print_cube(__gm__ float *gm_a, __gm__ float *gm_b,
                                         int32_t c2v_consumer_buf)
{
#if defined(__DAV_CUBE__)
    int64_t base0 = 0;
    int64_t base1024 = 1024;

    auto pipe = TPipe<0, FIFOType::VEC_FIFO, 8, 8,
                      Tile<TileType::Acc, float, 16, 16, BLayout::ColMajor, 16, 16,
                           SLayout::RowMajor, 1024, PadValue::Null>,
                      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, 8, 16,
                           SLayout::NoneBox, 512, PadValue::Null>>(c2v_consumer_buf);
    Tile<TileType::Mat, float, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null> matA;
    TASSIGN(matA, base0);
    Tile<TileType::Mat, float, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null> matB;
    TASSIGN(matB, base1024);

    Tile<TileType::Left, float, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null> leftA;
    TASSIGN(leftA, base0);
    Tile<TileType::Right, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor, 512, PadValue::Null> rightB;
    TASSIGN(rightB, base0);

    Tile<TileType::Acc, float, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 1024, PadValue::Null> accC;
    TASSIGN(accC, base0);

    using GTShape = pto::Shape<1, 1, 1, 16, 16>;
    using GTStride = pto::Stride<256, 256, 256, 16, 1>;
    using GlobalFloat = GlobalTensor<float, GTShape, GTStride, pto::Layout::ND>;

    GTShape shape = GTShape();
    GTStride stride = GTStride();

    GlobalFloat gA(gm_a, shape, stride);
    GlobalFloat gB(gm_b, shape, stride);

    TLOAD(matA, gA);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TLOAD(matB, gB);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(leftA, matA);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TMOV(rightB, matB);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(accC, leftA, rightB);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TPUSH(accC, pipe);
    pipe_barrier(PIPE_ALL);
#endif
}

static void matmul_tpush_tpop_print_vector(int32_t c2v_consumer_buf)
{
#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);

    int64_t base0 = 0;

    auto pipe = TPipe<0, FIFOType::VEC_FIFO, 8, 8,
                      Tile<TileType::Acc, float, 16, 16, BLayout::ColMajor, 16, 16,
                           SLayout::RowMajor, 1024, PadValue::Null>,
                      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, 8, 16,
                           SLayout::NoneBox, 512, PadValue::Null>>(c2v_consumer_buf);
    Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, 8, 16, SLayout::NoneBox, 512, PadValue::Null> vecPrint;
    TASSIGN(vecPrint, base0);
    Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, 8, 16, SLayout::NoneBox, 512, PadValue::Null> fifoTile;

    TPOP(fifoTile, pipe);
    TMOV(vecPrint, fifoTile);
    TPRINT(vecPrint);
    TFREE(pipe);
    pipe_barrier(PIPE_ALL);
#endif
}

__global__ AICORE void matmul_tpush_tpop_print(__gm__ float *gm_a, __gm__ float *gm_b,
                                               int32_t c2v_consumer_buf)
{
    matmul_tpush_tpop_print_cube(gm_a, gm_b, c2v_consumer_buf);
    matmul_tpush_tpop_print_vector(c2v_consumer_buf);
}

void LaunchMatmulTPushPopPrint(uint8_t *a, uint8_t *b, int32_t c2vBuf, void *stream)
{
    matmul_tpush_tpop_print<<<1, nullptr, stream>>>(reinterpret_cast<float *>(a),
                                                    reinterpret_cast<float *>(b),
                                                    c2vBuf);
}
