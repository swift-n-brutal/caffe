#include <vector>

#include "caffe/layers/select_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SelectForward(const int n_threads, const Dtype* bottom_data,
    const Dtype* select_data, Dtype* top_data,
    const int cand_id, const int inner_dim) {
  CUDA_KERNEL_LOOP(index, n_threads) {
    const int i = index / inner_dim;
    const int select_id = static_cast<int>(select_data[i]);
//    DCHECK_GE(select_id, 0);
//    DCHECK_LT(select_id, num_cand);
    if (cand_id == select_id) { top_data[index] = bottom_data[index]; }
  }
}

template <typename Dtype>
void SelectLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* select_data = bottom[num_cand_]->gpu_data();
  const int n_threads = top[0]->count();
  for (int i = 0; i < num_cand_; ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    SelectForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
        n_threads, bottom_data, select_data, top_data,
        i, inner_dim_);
  }
}

template <typename Dtype>
__global__ void SelectBackward(const int n_threads, const Dtype* top_diff,
    const Dtype* select_data, Dtype* bottom_diff,
    const int cand_id, const int inner_dim) {
  CUDA_KERNEL_LOOP(index, n_threads) {
    const int i = index / inner_dim;
    const int select_id = static_cast<int>(select_data[i]);
    if (cand_id == select_id) {
      bottom_diff[index] = top_diff[index];
    } else {
      bottom_diff[index] = 0;
    }
  }
}

template <typename Dtype>
void SelectLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[num_cand_]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to selection inputs.";
  }
//  if (!propagate_down[0]) { return; }
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* select_data = bottom[num_cand_]->gpu_data();
  const int n_threads = bottom[0]->count();
  for (int i = 0; i < num_cand_; ++i) {
    if (!propagate_down[i]) { continue; }
    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    SelectBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
        n_threads, top_diff, select_data, bottom_diff,
        i, inner_dim_);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SelectLayer);

}  // namespace caffe
