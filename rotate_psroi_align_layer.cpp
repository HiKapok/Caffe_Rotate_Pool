#include <cfloat>
#include <cmath>

#include "caffe/rotate_pool.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;


namespace caffe {

template <typename Dtype>
void RotatePSROIAlignLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  RotatePSROIAlignParameter rotate_psalign_param = this->layer_param_.rotate_psalign_param();
  CHECK_GT(rotate_psalign_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(rotate_psalign_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = rotate_psalign_param.pooled_h();
  pooled_width_ = rotate_psalign_param.pooled_w();
  spatial_scale_ = rotate_psalign_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void RotatePSROIAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), int(channels_ / (pooled_height_ * pooled_width_)), pooled_height_,
      pooled_width_);
  max_idx.Reshape(bottom[1]->num(), int(channels_ / (pooled_height_ * pooled_width_)), pooled_height_,
      pooled_width_);
}

template <typename Dtype>
void RotatePSROIAlignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  NOT_IMPLEMENTED;
}

template <typename Dtype>
void RotatePSROIAlignLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(RotatePSROIAlignLayer);
#endif

INSTANTIATE_CLASS(RotatePSROIAlignLayer);
REGISTER_LAYER_CLASS(RotatePSROIAlign);

}  // namespace caffe
