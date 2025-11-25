// Q, K, V 생성 커널 (타일링 적용)
// Local memory를 사용한 타일 기반 행렬 곱셈

#define TILE_SIZE 16

__kernel void compute_qkv_tiled(
    __global const float* input,           // [tokens, embed_dim]
    __global const float* weight,          // [3*embed_dim, embed_dim] (Q, K, V weights concatenated)
    __global const float* bias,            // [3*embed_dim] (Q, K, V bias concatenated)
    __global float* Q,                     // [tokens, embed_dim]
    __global float* K,                     // [tokens, embed_dim]
    __global float* V,                     // [tokens, embed_dim]
    const int tokens,
    const int embed_dim
) {
    // 각 work-item은 output의 한 요소를 계산
    int t = get_global_id(0);  // token index
    int d = get_global_id(1);  // dimension index

    if (t >= tokens || d >= embed_dim) return;

    // Local memory for tiling
    __local float input_tile[TILE_SIZE][TILE_SIZE + 1];  // +1 for avoiding bank conflicts
    __local float weight_tile[TILE_SIZE][TILE_SIZE + 1];

    int local_row = get_local_id(0);
    int local_col = get_local_id(1);

    // Q, K, V 각각에 대한 누적값
    float sum_q = bias[d];           // Q bias
    float sum_k = bias[embed_dim + d];    // K bias
    float sum_v = bias[2 * embed_dim + d]; // V bias

    // 타일 단위로 행렬 곱셈 수행
    int num_tiles = (embed_dim + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < num_tiles; tile++) {
        // Load input tile into local memory
        int input_col = tile * TILE_SIZE + local_col;
        if (t < tokens && input_col < embed_dim) {
            input_tile[local_row][local_col] = input[t * embed_dim + input_col];
        } else {
            input_tile[local_row][local_col] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Process Q
        int weight_col = tile * TILE_SIZE + local_col;
        if (d < embed_dim && weight_col < embed_dim) {
            weight_tile[local_row][local_col] = weight[d * embed_dim + weight_col];
        } else {
            weight_tile[local_row][local_col] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            sum_q += input_tile[local_row][k] * weight_tile[local_row][k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Process K
        if (d < embed_dim && weight_col < embed_dim) {
            weight_tile[local_row][local_col] = weight[(embed_dim + d) * embed_dim + weight_col];
        } else {
            weight_tile[local_row][local_col] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            sum_k += input_tile[local_row][k] * weight_tile[local_row][k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Process V
        if (d < embed_dim && weight_col < embed_dim) {
            weight_tile[local_row][local_col] = weight[(2 * embed_dim + d) * embed_dim + weight_col];
        } else {
            weight_tile[local_row][local_col] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            sum_v += input_tile[local_row][k] * weight_tile[local_row][k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write results
    Q[t * embed_dim + d] = sum_q;
    K[t * embed_dim + d] = sum_k;
    V[t * embed_dim + d] = sum_v;
}

