#include <vector>

#include "caffe/layers/euclidean_dist_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanDistLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  temp_.ReshapeLike(*bottom[0]);
  top[0]->Reshape(bottom[0]->num(), 1, 1, 1);
  sum_multiplier_.Reshape(bottom[0]->count(1), 1, 1, 1);
  caffe_set(sum_multiplier_.count(), Dtype(1), sum_multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void EuclideanDistLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = count / num;
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  caffe_powx(count, diff_.cpu_data(), Dtype(2), temp_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, Dtype(0.5), temp_.cpu_data(),
      sum_multiplier_.cpu_data(), 0.,
      top[0]->mutable_cpu_data());
}

template <typename Dtype>
void EuclideanDistLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = sum_multiplier_.count();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1,
      1., top[0]->cpu_diff(), sum_multiplier_.cpu_data(),
      0., temp_.mutable_cpu_data());
  caffe_mul(count, temp_.cpu_data(), diff_.cpu_data(), diff_.mutable_cpu_data());
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          sign,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanDistLayer);
#endif

INSTANTIATE_CLASS(EuclideanDistLayer);
REGISTER_LAYER_CLASS(EuclideanDist);

}  // namespace caffe
