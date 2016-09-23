#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_dist_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxDistForwardGPU(const int n_threads,
          const Dtype* prob_data, const Dtype* label, Dtype* dist,
          const int dim, const int inner_num) {
  CUDA_KERNEL_LOOP(index, n_threads) {
    const int i = index / inner_num;
    const int j = index % inner_num;
    const int label_value = static_cast<int>(label[index]);
    dist[index] = -log(max(prob_data[i * dim + label_value * inner_num + j],
                      Dtype(FLT_MIN))) / inner_num;
  }
}

template <typename Dtype>
void SoftmaxWithDistLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  int dim = prob_.count() / outer_num_;
  const int n_threads = outer_num_ * inner_num_;
  Dtype* dist = top[0]->mutable_gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxDistForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(n_threads),
      CAFFE_CUDA_NUM_THREADS>>>(n_threads, prob_data, label, dist,
      dim, inner_num_);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void SoftmaxDistBackwardGPU(const int n_threads,
    const Dtype* prob_data, const Dtype* label,
    const Dtype* top_diff, Dtype* bottom_diff,
    const int dim, const int inner_num, const int num_softmax) {
  CUDA_KERNEL_LOOP(index, n_threads) {
    const int i = index / dim;
    const int j = index % inner_num;
    const int c = (index / inner_num) % num_softmax;
    const int label_value = static_cast<int>(label[i*inner_num + j]);
    bottom_diff[index] = (prob_data[index] - ((c == label_value) ? 1. : 0.))
        * top_diff[i*inner_num + j] / inner_num;
  }
}

template <typename Dtype>
void SoftmaxWithDistLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    int dim = prob_.count() / outer_num_;
    const int num_softmax = bottom[0]->shape(softmax_axis_);
    const int n_threads = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxDistBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(n_threads),
      CAFFE_CUDA_NUM_THREADS>>>(n_threads, prob_data, label,
      top_diff, bottom_diff, dim, inner_num_, num_softmax);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithDistLayer);

}  // namespace caffe
