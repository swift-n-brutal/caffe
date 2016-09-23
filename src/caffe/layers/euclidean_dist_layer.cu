#include <vector>

#include "caffe/layers/euclidean_dist_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanDistLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = count / num;
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  caffe_gpu_powx(count, diff_.gpu_data(), Dtype(2), temp_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, Dtype(0.5), temp_.gpu_data(),
      sum_multiplier_.gpu_data(), 0.,
      top[0]->mutable_gpu_data());
}

template <typename Dtype>
void EuclideanDistLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = sum_multiplier_.count();
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1,
      1., top[0]->gpu_diff(), sum_multiplier_.gpu_data(),
      0., temp_.mutable_gpu_data());
  caffe_gpu_mul(count, temp_.gpu_data(), diff_.gpu_data(), diff_.mutable_gpu_data());
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          sign,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanDistLayer);

}  // namespace caffe
