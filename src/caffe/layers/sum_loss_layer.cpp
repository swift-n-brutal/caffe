#include <vector>

#include "caffe/layers/sum_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SumLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  vector<int> loss_shape(0);
  top[0]->Reshape(loss_shape);
  sum_multiplier_.Reshape(count, 1, 1, 1);
  caffe_set(count, Dtype(1), sum_multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void SumLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  Dtype dot = caffe_cpu_dot(count, sum_multiplier_.cpu_data(), bottom[0]->cpu_data());
  Dtype loss = dot / num;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void SumLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
    caffe_cpu_axpby(
        bottom[0]->count(),              // count
        alpha,                              // alpha
        sum_multiplier_.cpu_data(),                   // a
        Dtype(0),                           // beta
        bottom[0]->mutable_cpu_diff());  // b
  }
}

#ifdef CPU_ONLY
STUB_GPU(SumLossLayer);
#endif

INSTANTIATE_CLASS(SumLossLayer);
REGISTER_LAYER_CLASS(SumLoss);

}  // namespace caffe
