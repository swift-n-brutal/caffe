#include <vector>

#include "caffe/layers/select_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SelectLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  num_cand_ = bottom.size() - 1;
  outer_dim_ = bottom[num_cand_]->num();
  CHECK_EQ(outer_dim_, bottom[num_cand_]->count());
  inner_dim_ = bottom[0]->count(1);
  int count = outer_dim_ * inner_dim_;
  for (int i = 0; i < num_cand_; ++i) {
    CHECK_EQ(outer_dim_, bottom[i]->num())
      << "Each blob should have the same batch size.";
    CHECK_EQ(count, bottom[i]->count())
      << "Each blob should have the same count.";
  }
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SelectLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* select_data = bottom[num_cand_]->cpu_data();
  for (int i = 0; i < outer_dim_; ++i) {
    const int index = static_cast<int>(select_data[i]);
    DCHECK_GE(index, 0);
    DCHECK_LT(index, num_cand_);
    caffe_copy(inner_dim_, bottom[index]->cpu_data() + inner_dim_*i, top_data);
    top_data += inner_dim_;
  }
}

template <typename Dtype>
void SelectLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[num_cand_]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to selection inputs";
  }
//  if (!propagate_down[0]) { return; }
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* select_data = bottom[num_cand_]->cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < num_cand_; ++i) {
    caffe_set(count, Dtype(0), bottom[0]->mutable_cpu_diff());
  }
  for (int i = 0; i < outer_dim_; ++i) {
    const int index = static_cast<int>(select_data[i]);
    if (propagate_down[index]) {
      caffe_copy(inner_dim_, top_diff, bottom[index]->mutable_cpu_data() + inner_dim_*i);
    }
    top_diff += inner_dim_;
  }
}

#ifdef CPU_ONLY
STUB_GPU(SelectLayer);
#endif

INSTANTIATE_CLASS(SelectLayer);
REGISTER_LAYER_CLASS(Select);

}  // namespace caffe
