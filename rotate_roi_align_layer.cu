#include <cfloat>
#include <cmath>
#include <stdint.h>
#include <stdio.h>
//#include <cstdio>

#include "caffe/rotate_pool.hpp"
#include "stdio.h"
#include "caffe/util/gpu_util.cuh"
using std::max;
using std::min;



// For atomicAdd.
// __device__ inline double atomicAdd(double* address, double val)
// {
//     unsigned long long int* address_as_ull = (unsigned long long int*)address;
//     unsigned long long int old = *address_as_ull, assumed;
//     do {
//         assumed = old;
//         old = atomicCAS(address_as_ull, assumed,
//                 __double_as_longlong(val + __longlong_as_double(assumed)));
//     } while (assumed != old);
//     return __longlong_as_double(old);
// }

// Custom implementation of atomicAdd for double.
// This implementation is copied from CUDA manual.
// CUDA_ATOMIC_WRAPPER(Add, double) {
//   uint64_t* address_as_ull = (uint64_t*)address;
//   uint64_t old = *address_as_ull, assumed;

//   do {
//     assumed = old;
//     old = atomicCAS(address_as_ull, assumed,
//                     __double_as_longlong(val + __longlong_as_double(assumed)));

//     // Note: uses integer comparison to avoid hang in case of NaN
//   } while (assumed != old);

//   return __longlong_as_double(old);
// }

#define F_DEVPTR(ptr) ((float*)((ptr)->desc.dev_ptr))

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

namespace caffe {

__device__ static float ROIAlignGetCoeff(float dh, float dw){
     dw = dw > 0 ? dw : -dw;
     dh = dh > 0 ? dh : -dh;
     return (1.0f - dh) * (1.0f - dw);
}

/**
  * Implementation of the bilinear interpolation.
  */
template <typename Dtype>
__device__ static float ROIAlignGetInterpolating(const Dtype* data, const float h,
  const float w, const int height, const int width){
    float retVal = 0.0f;
    int h1 = floorf(h);
    int w1 = floorf(w);
    bool overflow = (h1<0) || (w1<0) || (h1 >=height) || (w1>=width);
    retVal += overflow? 0.0f : data[(h1 * width + w1)] * ROIAlignGetCoeff(h - float(h1), w - float(w1));
    h1 = ceilf(h);
    w1 = floorf(w);
    overflow = (h1<0) || (w1<0) || (h1 >=height) || (w1>=width);
    retVal += overflow? 0.0f : data[(h1 * width + w1)] * ROIAlignGetCoeff(h - float(h1), w - float(w1));
    h1 = floorf(h);
    w1 = ceilf(w);
    overflow = (h1<0) || (w1<0) || (h1 >=height) || (w1>=width);
    retVal += overflow? 0.0f : data[(h1 * width + w1)] * ROIAlignGetCoeff(h - float(h1), w - float(w1));
    h1 = ceilf(h);
    w1 = ceilf(w);
    overflow = (h1<0) || (w1<0) || (h1 >=height) || (w1>=width);
    retVal += overflow? 0.0f : data[(h1 * width + w1)] * ROIAlignGetCoeff(h - float(h1), w - float(w1));
    return retVal;
}
/**
  * Get the derivative of the bilinear interpolation.
  */
template <typename Dtype>
__device__ static void ROIAlignDistributeDiff(Dtype* diff, const Dtype top_diff,
  const float h, const float w, const int height, const int width){
    int h1 = floorf(h);
    int w1 = floorf(w);
    bool overflow = (h1<0) || (w1<0) || (h1 >=height) || (w1>=width);
    if (!overflow){
      caffe_gpu_atomic_add(
        top_diff * ROIAlignGetCoeff(h - float(h1), w - float(w1)), diff + (h1 * width + w1));
    }
    h1 = ceilf(h);
    w1 = floorf(w);
    overflow = (h1<0) || (w1<0) || (h1 >=height) || (w1>=width);
    if (!overflow){
      caffe_gpu_atomic_add(
        top_diff * ROIAlignGetCoeff(h - float(h1), w - float(w1)), diff + (h1 * width + w1));
    }
    h1 = floorf(h);
    w1 = ceilf(w);
    overflow = (h1<0) || (w1<0) || (h1 >=height) || (w1>=width);
    if (!overflow){
      caffe_gpu_atomic_add(
        top_diff * ROIAlignGetCoeff(h - float(h1), w - float(w1)), diff + (h1 * width + w1));
    }
    h1 = ceilf(h);
    w1 = ceilf(w);
    overflow = (h1<0) || (w1<0) || (h1 >=height) || (w1>=width);
    if (!overflow){
      caffe_gpu_atomic_add(
        top_diff * ROIAlignGetCoeff(h - float(h1), w - float(w1)), diff + (h1 * width + w1));
    }
}

template <typename Dtype>
__global__ void RoiAlignForward(const int nthreads, const Dtype* bottom_data_start,
    const double spatial_scale, const int height, const int width,
    const int channels, const int pooled_height, const int pooled_width,
    const int sample_height, const int sample_width,
    const Dtype* bottom_rois_start, Dtype* top_data, int* argmax_data, const double pi)
{
  CUDA_1D_KERNEL_LOOP(index, nthreads)
  {
    // (n, ph, pw, c) is an element in the pooled output
    int n = index;
    int pw = n % pooled_width;
    n /= pooled_width;
    int ph = n % pooled_height;
    n /= pooled_height;
    int c = n % channels;
    n /= channels;

    const Dtype* bottom_rois = bottom_rois_start + n * 6;
    //bottom_rois += n * 6;
    //bottom_rois += n * 6;
    int roi_batch_ind = bottom_rois[0];

    //ps align max pooling
    const Dtype* bottom_data = bottom_data_start + roi_batch_ind * channels * height * width;
    //bottom_data += roi_batch_ind * channels * height * width;

    Dtype roi_ctr_x = static_cast<Dtype>(bottom_rois[1]) * spatial_scale;
    Dtype roi_ctr_y = static_cast<Dtype>(bottom_rois[2]) * spatial_scale;
    Dtype roi_h = static_cast<Dtype>(bottom_rois[3]) * spatial_scale;
    Dtype roi_w = static_cast<Dtype>(bottom_rois[4]) * spatial_scale;
    Dtype roi_angle = static_cast<Dtype>(bottom_rois[5]);

    if (roi_angle < 0) roi_angle += 180;
    if (roi_angle > 180) roi_angle -= 180;
    Dtype angle = roi_angle;//anchor_rect.size.width >= anchor_rect.size.height ? -anchor_rect.angle : 90 - anchor_rect.angle;
    angle = angle / 180 * pi;
    Dtype shorter_side = roi_h;// std::min(width, height);
    Dtype longer_side = roi_w;// std::max(width, height);

    const float grid_size_x = shorter_side / pooled_width;
    const float grid_size_y = longer_side / pooled_height;

    float grid_start_y = (pooled_height - 1 - ph) * grid_size_y;
    float grid_end_y = ((pooled_height - 1 - ph) + 1)* grid_size_y;
    float grid_start_x = pw * grid_size_x;
    float grid_end_x = (pw + 1)* grid_size_x;

    float tmp = float(-1e20);
    float tmp2;
    int buf_value = -1;
    for (int32_t sample_y = 1; sample_y < sample_height + 1; ++sample_y)
    {
        for (int32_t sample_x = 1; sample_x < sample_width + 1; ++sample_x)
        {
            float final_x = grid_start_x + sample_x * grid_size_x / (sample_width + 1) - shorter_side / 2.f;
            float final_y = grid_start_y + sample_y * grid_size_y / (sample_height + 1) - longer_side / 2.f;

            tmp2 = ROIAlignGetInterpolating(bottom_data + c * height * width, roi_ctr_y + final_x * std::cos(angle) - final_y * std::sin(angle), roi_ctr_x + final_x * std::sin(angle) + final_y * std::cos(angle), height, width);
            if (tmp2 > tmp){
                tmp = tmp2;
                buf_value = sample_x + sample_y * (sample_width + 1);
            }

        }
    }

    top_data[index] = tmp;
    argmax_data[index] = buf_value;
  }
}

template <typename Dtype>
void RoiAlignForwardLaucher(
    const Dtype* bottom_data, const float spatial_scale, const int num_rois,
    const int height, const int width, const int channels,
    const int pooled_height, const int pooled_width,
    const int sample_height, const int sample_width,
    const Dtype* bottom_rois, Dtype* top_data, int* argmax_data) {
  const int kThreadsPerBlock = CAFFE_CUDA_NUM_THREADS;
  const int output_size = num_rois * pooled_height * pooled_width * channels;
  cudaError_t err;

  const double pi = std::acos(-1);


  RoiAlignForward<<<CAFFE_GET_BLOCKS(output_size), CAFFE_CUDA_NUM_THREADS>>>(
      output_size, bottom_data, spatial_scale, height, width, channels,
      pooled_height, pooled_width, sample_height, sample_width,
      bottom_rois, top_data, argmax_data, pi);


  // RoiAlignForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
  //                      kThreadsPerBlock, 0>>>(
  //     output_size, bottom_data, spatial_scale, height, width, channels,
  //     pooled_height, pooled_width, sample_height, sample_width,
  //     bottom_rois, top_data, argmax_data, pi);
  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }
}


template <typename Dtype>
__global__ void RoiAlignBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const double spatial_scale,
    const int height, const int width, const int channels,
    const int pooled_height, const int pooled_width,
    const int sample_height, const int sample_width,
    Dtype* bottom_diff_start,
    const Dtype* bottom_rois_start, const double pi) {
  CUDA_1D_KERNEL_LOOP(index, nthreads)
  {
    // (n, h, w, c) coords in bottom
    int n = index;
    int pw = n % pooled_width;
    n /= pooled_width;
    int ph = n % pooled_height;
    n /= pooled_height;
    int c = n % channels;
    n /= channels;

    const Dtype* bottom_rois = bottom_rois_start + n * 6;
    //bottom_rois += n * 6;
    int roi_batch_ind = bottom_rois[0];

    //ps align max pooling
    Dtype* bottom_diff = bottom_diff_start + roi_batch_ind * channels * height * width;
    //bottom_diff += roi_batch_ind * channels * height * width;

    Dtype roi_ctr_x = static_cast<Dtype>(bottom_rois[1]) * spatial_scale;
    Dtype roi_ctr_y = static_cast<Dtype>(bottom_rois[2]) * spatial_scale;
    Dtype roi_h = static_cast<Dtype>(bottom_rois[3]) * spatial_scale;
    Dtype roi_w = static_cast<Dtype>(bottom_rois[4]) * spatial_scale;
    Dtype roi_angle = static_cast<Dtype>(bottom_rois[5]);

    if (roi_angle < 0) roi_angle += 180;
    if (roi_angle > 180) roi_angle -= 180;
    Dtype angle = roi_angle;//anchor_rect.size.width >= anchor_rect.size.height ? -anchor_rect.angle : 90 - anchor_rect.angle;
    //const double pi = std::acos(-1);
    angle = angle / 180 * pi;
    Dtype shorter_side = roi_h;// std::min(width, height);
    Dtype longer_side = roi_w;// std::max(width, height);

    const float grid_size_x = shorter_side / pooled_width;
    const float grid_size_y = longer_side / pooled_height;

    float grid_start_y = (pooled_height - 1 - ph) * grid_size_y;
    float grid_end_y = ((pooled_height - 1 - ph) + 1)* grid_size_y;
    float grid_start_x = pw * grid_size_x;
    float grid_end_x = (pw + 1)* grid_size_x;

    Dtype tmp = top_diff[index];
    int buffer_value = argmax_data[index];
    int sample_x = buffer_value % (sample_width + 1);
    int sample_y = buffer_value / (sample_width + 1);
    float final_x = grid_start_x + sample_x * grid_size_x / (sample_width + 1) - shorter_side / 2.f;
    float final_y = grid_start_y + sample_y * grid_size_y / (sample_height + 1) - longer_side / 2.f;

    ROIAlignDistributeDiff(bottom_diff + c * height * width, tmp, roi_ctr_y + final_x * std::cos(angle) - final_y * std::sin(angle), roi_ctr_x + final_x * std::sin(angle) + final_y * std::cos(angle), height, width);
  }
}

template <typename Dtype>
void RoiAlignBackwardLaucher(const Dtype* top_diff, const float spatial_scale,
  const int batch_size, const int num_rois,
    const int height, const int width, const int channels,
    const int pooled_height, const int pooled_width,
    const int sample_height, const int sample_width,
    const Dtype* bottom_rois, Dtype* bottom_diff,
    const int* argmax_data)
{
  const int kThreadsPerBlock = CAFFE_CUDA_NUM_THREADS;
  const int output_size = num_rois * pooled_height * pooled_width * channels;
  const int bottom_count = batch_size * height * width * channels;
  cudaError_t err;

  //cudaMemsetAsync(bottom_diff, 0, sizeof(float) * bottom_count, d.stream());
  //cudaMemsetAsync(F_DEVPTR(bottom_diff), 0, sizeof(float) * bottom_count, d.stream());
  const double pi = std::acos(-1);


  RoiAlignBackward<<<CAFFE_GET_BLOCKS(output_size), CAFFE_CUDA_NUM_THREADS>>>(
      output_size, top_diff, argmax_data, num_rois, spatial_scale,
      height, width, channels, pooled_height, pooled_width,
      sample_height, sample_width, bottom_diff, bottom_rois, pi);

  // RoiAlignBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
  //                     kThreadsPerBlock, 0>>>(
  //     output_size, top_diff, argmax_data, num_rois, spatial_scale,
  //     height, width, channels, pooled_height, pooled_width,
  //     sample_height, sample_width, bottom_diff, bottom_rois, pi);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

}

template <typename Dtype>
void RotateROIAlignLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  RoiAlignForwardLaucher(bottom_data, spatial_scale_, bottom[1]->num(), height_, width_, channels_, pooled_height_, pooled_width_, 2, 2, bottom_rois, top_data, max_idx.mutable_gpu_data());

}


template <typename Dtype>
void RotateROIAlignLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  //std::cout<<top_diff[0]<<std::endl;
  Dtype* backbone_diff = bottom[0]->mutable_gpu_diff();
  Dtype* proposal_diff = bottom[1]->mutable_gpu_diff();

  //const int count = bottom[0]->count();
  const int backbone_count = bottom[0]->count();
  const int proposal_count = bottom[1]->count();

  //caffe_gpu_set(count, Dtype(0.), backbone_diff);
  caffe_gpu_set(backbone_count, Dtype(0.), backbone_diff);
  caffe_gpu_set(proposal_count, Dtype(0.), proposal_diff);


  RoiAlignBackwardLaucher(top_diff, spatial_scale_, bottom[0]->num(), bottom[1]->num(), height_, width_, channels_, pooled_height_, pooled_width_, 2, 2, bottom_rois, backbone_diff, max_idx.gpu_data());
}


INSTANTIATE_LAYER_GPU_FUNCS(RotateMyROIAlignLayer);

}  // namespace caffe
