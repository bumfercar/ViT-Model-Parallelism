#pragma warning(disable : 4996)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include "Network.h"

////////////////////////////////////// OpenCL Global Variables //////////////////////////////////////
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel_qkv_tiled;
#define img_size 224
#define patch_size 16
#define in_chans 3
#define num_classes 1000
#define embed_dim 768
#define depth 12
#define num_heads 12
#define mlp_ratio 4.0
#define dropout 0.0
#define attn_dropout 0.0
#define drop_path_rate 0.0
#define eps 1e-6

////////////////////////////////////// OpenCL Helper Functions //////////////////////////////////////

// 파일에서 커널 소스 읽기
char* read_kernel_source(const char* filename, size_t* size) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open kernel file: %s\n", filename);
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    rewind(fp);

    char* source = (char*)malloc(*size + 1);
    fread(source, 1, *size, fp);
    source[*size] = '\0';
    fclose(fp);
    return source;
}

// OpenCL 초기화
void init_opencl() {
    cl_int err;

    // Platform 선택
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to get platform IDs: %d\n", err);
        exit(1);
    }

    // Device 선택 (GPU 우선)
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        // GPU가 없으면 CPU 사용
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to get device IDs: %d\n", err);
            exit(1);
        }
    }

    // Context 생성
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create context: %d\n", err);
        exit(1);
    }

    // Command queue 생성
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create command queue: %d\n", err);
        exit(1);
    }

    // 커널 소스 읽기
    size_t source_size;
    char* source = read_kernel_source("qkv_kernel.cl", &source_size);

    // Program 생성 및 빌드
    program = clCreateProgramWithSource(context, 1, (const char**)&source, &source_size, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create program: %d\n", err);
        exit(1);
    }

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to build program: %d\n", err);

        // 빌드 로그 출력
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        exit(1);
    }

    free(source);

    // Kernel 생성
    kernel_qkv_tiled = clCreateKernel(program, "compute_qkv_tiled", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create kernel_qkv_tiled: %d\n", err);
        exit(1);
    }

    printf("OpenCL initialized successfully\n");
}

// OpenCL 정리
void cleanup_opencl() {
    clReleaseKernel(kernel_qkv_tiled);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

////////////////////////////////////// ViT function //////////////////////////////////////

void Conv2d(float* input, float* output, Network weight, Network bias) {
    int output_size = img_size / patch_size;

    for (int oc = 0; oc < embed_dim; ++oc) {
        for (int oh = 0; oh < output_size; ++oh) {
            for (int ow = 0; ow < output_size; ++ow) {
                float sum = bias.data[oc];

                for (int ic = 0; ic < in_chans; ++ic) {
                    for (int kh = 0; kh < patch_size; ++kh) {
                        for (int kw = 0; kw < patch_size; ++kw) {
                            int ih = oh * patch_size + kh;
                            int iw = ow * patch_size + kw;
                            int input_idx = (ic * img_size + ih) * img_size + iw;
                            int kernel_idx = ((oc * in_chans + ic) * patch_size + kh) * patch_size + kw;

                            sum += input[input_idx] * weight.data[kernel_idx];
                        }
                    }
                }

                output[(oc * output_size + oh) * output_size + ow] = sum;
            }
        }
    }
}

void flatten_transpose(float* input, float* output) {
    int output_size = img_size / patch_size;
    int num_patches = output_size * output_size;

    // �� ���� ��ġ(oh, ow)�� �ϳ��� ��ġ�� ����Ͽ� patch index ���
    for (int oh = 0; oh < output_size; oh++) {
        for (int ow = 0; ow < output_size; ow++) {
            int patch_idx = oh * output_size + ow;
            for (int oc = 0; oc < embed_dim; oc++) {
                // ���� �Է��� (oc, oh, ow)
                int idx_input = (oc * output_size + oh) * output_size + ow;
                // ���ϴ� ����� (patch_idx, oc)
                int idx_output = patch_idx * embed_dim + oc;
                output[idx_output] = input[idx_input];
                //printf("%f ",output[idx_output]);
            }
        }
    }
}

void class_token(float* patch_tokens, float* final_tokens, Network cls_tk) {
    // �̹����� ��ġ �� ���: output_size = img_size / patch_size, num_patches = output_size^2
    int output_size = img_size / patch_size;
    int num_patches = output_size * output_size;

    // 1. ù ��° ��ū�� class token ���� (networks[0].data�� �����, embed_dim ����)
    for (int j = 0; j < embed_dim; j++) {
        final_tokens[j] = cls_tk.data[j];
    }

    // 2. ���� patch_tokens�� �̾����
    // final_tokens�� �ε��� embed_dim����, patch_tokens ��ü(embed_dim * num_patches) ����
    memcpy(final_tokens + embed_dim, patch_tokens, sizeof(float) * embed_dim * num_patches);

    int total_tokens = num_patches + 1; // class token + patch tokens
    for (int i = 0; i < total_tokens * embed_dim; i++) {
        //("%f ", final_tokens[i]);
    }
    //printf("\n");
}

void pos_emb(float* input, float* output, Network pos_emb) {
    // output_size: �� ���� ��ġ ��, num_patches: ��ü ��ġ ��, total_tokens: class token + patch tokens
    int output_size = img_size / patch_size;
    int num_patches = output_size * output_size;
    int total_tokens = num_patches + 1;
    int total_elements = total_tokens * embed_dim;
    for (int i = 0; i < total_elements; i++) {
        output[i] = input[i] + pos_emb.data[i];
    }
}

void layer_norm(float* input, float* output, Network weight, Network bias) {
    int token = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    
    for (int t = 0; t < token; t++) {
        float sum = 0.0, sum_sq = 0.0;
        for (int i = 0; i < embed_dim; i++) {
            float val = input[t * embed_dim + i];
            sum += val;
            sum_sq += val * val;
        }
        float mean = sum / embed_dim;
        float var = sum_sq / embed_dim - mean * mean;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < embed_dim; i++) {
            int idx = t * embed_dim + i;
            output[idx] = (input[idx] - mean) * inv_std * weight.data[i] + bias.data[i];
        }
    }
}

void multihead_attn(float* input, float* output,
    Network in_weight, Network in_bias, Network out_weight, Network out_bias) {

    int head_dim = embed_dim / num_heads, tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
   
    /*Allocate Q, K, V : tokens * dim*/
    float* Q = (float*)malloc(sizeof(float) * tokens * embed_dim);
    float* K = (float*)malloc(sizeof(float) * tokens * embed_dim);
    float* V = (float*)malloc(sizeof(float) * tokens * embed_dim);

    /*Q, K, V를 OpenCL로 병렬 생성 (타일링 적용)*/
    cl_int err;

    // OpenCL 버퍼 생성
    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(float) * tokens * embed_dim, input, &err);
    cl_mem d_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * 3 * embed_dim * embed_dim, in_weight.data, &err);
    cl_mem d_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * 3 * embed_dim, in_bias.data, &err);
    cl_mem d_Q = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                 sizeof(float) * tokens * embed_dim, NULL, &err);
    cl_mem d_K = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                 sizeof(float) * tokens * embed_dim, NULL, &err);
    cl_mem d_V = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                 sizeof(float) * tokens * embed_dim, NULL, &err);

    // 커널 인자 설정 (타일링 버전 사용)
    int embed_dim_arg = embed_dim;
    clSetKernelArg(kernel_qkv_tiled, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel_qkv_tiled, 1, sizeof(cl_mem), &d_weight);
    clSetKernelArg(kernel_qkv_tiled, 2, sizeof(cl_mem), &d_bias);
    clSetKernelArg(kernel_qkv_tiled, 3, sizeof(cl_mem), &d_Q);
    clSetKernelArg(kernel_qkv_tiled, 4, sizeof(cl_mem), &d_K);
    clSetKernelArg(kernel_qkv_tiled, 5, sizeof(cl_mem), &d_V);
    clSetKernelArg(kernel_qkv_tiled, 6, sizeof(int), &tokens);
    clSetKernelArg(kernel_qkv_tiled, 7, sizeof(int), &embed_dim_arg);

    // 커널 실행 (타일링 적용: local work size 설정)
    // tokens=197, embed_dim=768
    // global_size를 16의 배수로 올림
    size_t local_size[2] = {16, 16};  // TILE_SIZE와 동일
    size_t global_size[2] = {
        ((tokens + 15) / 16) * 16,     // 197 -> 208 (16의 배수)
        ((embed_dim + 15) / 16) * 16   // 768 -> 768 (이미 16의 배수)
    };
    err = clEnqueueNDRangeKernel(queue, kernel_qkv_tiled, 2, NULL, global_size, local_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to enqueue kernel: %d\n", err);
        exit(1);
    }

    // 결과 읽기
    clEnqueueReadBuffer(queue, d_Q, CL_TRUE, 0, sizeof(float) * tokens * embed_dim, Q, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_K, CL_TRUE, 0, sizeof(float) * tokens * embed_dim, K, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_V, CL_TRUE, 0, sizeof(float) * tokens * embed_dim, V, 0, NULL, NULL);

    // 버퍼 해제
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_weight);
    clReleaseMemObject(d_bias);
    clReleaseMemObject(d_Q);
    clReleaseMemObject(d_K);
    clReleaseMemObject(d_V);
    int print_tokens = tokens < 5 ? tokens : 5;
    int print_dims = embed_dim < 10 ? embed_dim : 10;

    /*Attn ����� ������ ����*/
    float* attn_output = (float*)malloc(sizeof(float) * tokens * embed_dim);
    for (int i = 0; i < tokens * embed_dim; i++) attn_output[i] = 0.0f;

    /*head���� attn ����*/
    for (int h = 0; h < num_heads; h++) {
        int head_offset = h * head_dim;

        // attn_score ���� ����
        float* scores = (float*)malloc(sizeof(float) * tokens * tokens);
        float* scores_tmp = (float*)malloc(sizeof(float) * tokens * tokens);


        // �� head�� ���� scaled-dot attn
        for (int i = 0; i < tokens; i++) {
            for (int j = 0; j < tokens; j++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    float q = Q[i * embed_dim + head_offset + d];
                    float k = K[j * embed_dim + head_offset + d];
                    score += q * k;
                }
                scores[i * tokens + j] = score / sqrtf((float)head_dim);
            }
        }

        // softmax ����
        for (int i = 0; i < tokens; i++) {
            float max_val = scores[i * tokens];
            for (int j = 1; j < tokens; j++) {
                if (scores[i * tokens + j] > max_val) max_val = scores[i * tokens + j];
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < tokens; j++) {
                scores[i * tokens + j] = expf(scores[i * tokens + j] - max_val);
                sum_exp += scores[i * tokens + j];
            }
            for (int j = 0; j < tokens; j++) {
                scores[i * tokens + j] /= sum_exp;
            }
        }

        // scores�� V�� ���� head output ���
        float* head_out = (float*)malloc(sizeof(float) * tokens * head_dim);
        for (int i = 0; i < tokens; i++) {
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int j = 0; j < tokens; j++) {
                    sum += scores[i * tokens + j] * V[j * embed_dim + head_offset + d];
                }
                head_out[i * head_dim + d] = sum;
            }
        }

        // head_out�� attn_output�� �ش� �κп� ����
        for (int i = 0; i < tokens; i++) {
            for (int d = 0; d < head_dim; d++) {
                attn_output[i * embed_dim + head_offset + d] = head_out[i * head_dim + d];
            }
        }

        free(scores);
        free(head_out);
    }

    free(Q); free(K); free(V);

    // ���� ���� ��������
    for (int t = 0; t < tokens; t++) {
        for (int i = 0; i < embed_dim; i++) {
            float sum = out_bias.data[i];
            for (int j = 0; j < embed_dim; j++) {
                sum += attn_output[t * embed_dim + j] * out_weight.data[i * embed_dim + j];
            }
            output[t * embed_dim + i] = sum;
        }
    }
    free(attn_output);
}

float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)));
}
void gelu_activation(float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = gelu(input[i]);
    }
}

void linear_layer(float* input, float* output, int tokens, int in_features, int out_features, Network weight, Network bias) {
    for (int t = 0; t < tokens; t++) {
        for (int o = 0; o < out_features; o++) {
            float sum = bias.data[o];
            for (int i = 0; i < in_features; i++) {
                sum += input[t * in_features + i] * weight.data[o * in_features + i];
            }
            output[t * out_features + o] = sum;
        }
    }
}
void mlp_block(float* input, float* output, Network fc1_weight, Network fc1_bias, Network fc2_weight, Network fc2_bias) {
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1; //197
    int Embed_dim = embed_dim; //768
    int hidden_dim = ((int)(embed_dim * mlp_ratio)); //3072



    float* fc1_out = (float*)malloc(sizeof(float) * tokens * hidden_dim);

    linear_layer(input, fc1_out, tokens, embed_dim, hidden_dim, fc1_weight, fc1_bias);
    // GELU Ȱ��ȭ
    for (int i = 0; i < tokens * hidden_dim; i++) {
        fc1_out[i] = gelu(fc1_out[i]);
    }
    // fc2: (tokens, in_dim)
    linear_layer(fc1_out, output, tokens, hidden_dim, embed_dim, fc2_weight, fc2_bias);
    free(fc1_out);
}

////////////////////////////////////// Encoder Architecture //////////////////////////////////////
void Encoder(float* input, float* output,
    Network ln1_w, Network ln1_b, Network attn_w, Network attn_b, Network attn_out_w, Network attn_out_b,
    Network ln2_w, Network ln2_b, Network mlp1_w, Network mlp1_b, Network mlp2_w, Network mlp2_b) {
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    float* ln1_out = (float*)malloc(sizeof(float) * tokens * embed_dim);
    float* attn_out = (float*)malloc(sizeof(float) * tokens * embed_dim);
    float* residual = (float*)malloc(sizeof(float) * tokens * embed_dim);
    float* ln2_out = (float*)malloc(sizeof(float) * tokens * embed_dim);
    float* mlp_out = (float*)malloc(sizeof(float) * tokens * embed_dim);

    /*LN1*/
    layer_norm(input, ln1_out, ln1_w, ln1_b);

    /*Attn*/
    multihead_attn(ln1_out, attn_out, attn_w, attn_b, attn_out_w, attn_out_b);

    /*Residual1*/
    for (int i = 0; i < tokens * embed_dim; i++) {
        residual[i] = input[i] + attn_out[i];
    }

    /*LN2*/
    layer_norm(residual, ln2_out, ln2_w, ln2_b);

    /*MLP*/
    mlp_block(ln2_out, mlp_out, mlp1_w, mlp1_b, mlp2_w, mlp2_b);

    /*Residual2*/
    for (int i = 0; i < tokens * embed_dim; i++) {
        output[i] = residual[i] + mlp_out[i];
    }

    free(ln1_out); free(attn_out); free(residual); free(ln2_out); free(mlp_out);
}

void Softmax(float* logits, float* probabilities, int length) {
    // ��ġ �������� ���� �ִ밪 ���
    float max_val = logits[0];
    for (int i = 1; i < length; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }

    // �� ���ҿ� ���� exp(logit - max_val)�� ����ϰ� �ջ�
    float sum_exp = 0.0f;
    for (int i = 0; i < length; i++) {
        probabilities[i] = expf(logits[i] - max_val);
        sum_exp += probabilities[i];
    }

    // Ȯ�������� ����ȭ
    for (int i = 0; i < length; i++) {
        probabilities[i] /= sum_exp;
    }
}

////////////////////////////////////// layer�� size //////////////////////////////////////
const int size[] = {
    embed_dim * (img_size / patch_size) * (img_size / patch_size), // conv2D
    embed_dim * (img_size / patch_size) * (img_size / patch_size), // flatten and transpose
    embed_dim * ((img_size / patch_size) * (img_size / patch_size) + 1), // class token
    embed_dim * ((img_size / patch_size) * (img_size / patch_size) + 1) // position embedding
};

const int enc_size = embed_dim * ((img_size / patch_size) * (img_size / patch_size) + 1);

////////////////////////////////////// Model Architecture //////////////////////////////////////
void ViT_seq(ImageData* image, Network* networks, float** probabilities) {

    int token_size = ((img_size / patch_size) * (img_size / patch_size) + 1);
    float* layer[4];
    float* enc_layer[12];
    float* enc_output;
    int  hidden_dim = ((int)(embed_dim * mlp_ratio));
    //printf("%d %d = %d\n", token_size, hidden_dim, token_size * hidden_dim);

    for (int i = 0; i < 4; i++) {
        layer[i] = (float*)malloc(sizeof(float) * size[i]);
    }
    for (int i = 0; i < 12; i++) {
        enc_layer[i] = (float*)malloc(sizeof(float) * enc_size);
    }
    enc_output = (float*)malloc(sizeof(float) * enc_size);

    for (int i = 0; i < image->n; i++) {
        /*patch embedding*/
        Conv2d(image[i].data, layer[0], networks[1], networks[2]);
        /*flatten and transpose*/
        flatten_transpose(layer[0], layer[1]);
        /*prepend class token*/
        class_token(layer[1], layer[2], networks[0]);
        /*position embedding*/
        pos_emb(layer[2], layer[3], networks[3]);

        /*Encoder - 12 Layers*/
        Encoder(layer[3], enc_layer[0],
            networks[4], networks[5], networks[6], networks[7],
            networks[8], networks[9], networks[10], networks[11],
            networks[12], networks[13], networks[14], networks[15]);

        Encoder(enc_layer[0], enc_layer[1],
            networks[16], networks[17], networks[18], networks[19],
            networks[20], networks[21], networks[22], networks[23],
            networks[24], networks[25], networks[26], networks[27]);

        Encoder(enc_layer[1], enc_layer[2],
            networks[28], networks[29], networks[30], networks[31],
            networks[32], networks[33], networks[34], networks[35],
            networks[36], networks[37], networks[38], networks[39]);

        Encoder(enc_layer[2], enc_layer[3],
            networks[40], networks[41], networks[42], networks[43],
            networks[44], networks[45], networks[46], networks[47],
            networks[48], networks[49], networks[50], networks[51]);

        Encoder(enc_layer[3], enc_layer[4],
            networks[52], networks[53], networks[54], networks[55],
            networks[56], networks[57], networks[58], networks[59],
            networks[60], networks[61], networks[62], networks[63]);

        Encoder(enc_layer[4], enc_layer[5],
            networks[64], networks[65], networks[66], networks[67],
            networks[68], networks[69], networks[70], networks[71],
            networks[72], networks[73], networks[74], networks[75]);

        Encoder(enc_layer[5], enc_layer[6],
            networks[76], networks[77], networks[78], networks[79],
            networks[80], networks[81], networks[82], networks[83],
            networks[84], networks[85], networks[86], networks[87]);

        Encoder(enc_layer[6], enc_layer[7],
            networks[88], networks[89], networks[90], networks[91],
            networks[92], networks[93], networks[94], networks[95],
            networks[96], networks[97], networks[98], networks[99]);

        Encoder(enc_layer[7], enc_layer[8],
            networks[100], networks[101], networks[102], networks[103],
            networks[104], networks[105], networks[106], networks[107],
            networks[108], networks[109], networks[110], networks[111]);

        Encoder(enc_layer[8], enc_layer[9],
            networks[112], networks[113], networks[114], networks[115],
            networks[116], networks[117], networks[118], networks[119],
            networks[120], networks[121], networks[122], networks[123]);

        Encoder(enc_layer[9], enc_layer[10],
            networks[124], networks[125], networks[126], networks[127],
            networks[128], networks[129], networks[130], networks[131],
            networks[132], networks[133], networks[134], networks[135]);

        Encoder(enc_layer[10], enc_layer[11],
            networks[136], networks[137], networks[138], networks[139],
            networks[140], networks[141], networks[142], networks[143],
            networks[144], networks[145], networks[146], networks[147]);

        layer_norm(enc_layer[11], enc_output, networks[148], networks[149]);

        /* Token �� ���� */
        float* cls_token = (float*)malloc(sizeof(float) * embed_dim);
        float* cls_output = (float*)malloc(sizeof(float) * num_classes);
        memcpy(cls_token, enc_output, sizeof(float) * embed_dim);

        linear_layer(cls_token, cls_output, 1, embed_dim, num_classes, networks[150], networks[151]);
        /* Ȯ������ ���� */
        Softmax(cls_output, probabilities[i], num_classes);
    }
}