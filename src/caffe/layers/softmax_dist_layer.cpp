#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_dist_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithDistLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}

template <typename Dtype>
void SoftmaxWithDistLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  top[0]->Reshape(outer_num_, 1, 1, inner_num_);
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
  sum_multiplier_.Reshape(bottom[0]->shape(softmax_axis_), 1, 1, 1);
  caffe_set(sum_multiplier_.count(), Dtype(1), sum_multiplier_.mutable_cpu_data());
  temp_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
Dtype SoftmaxWithDistLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SoftmaxWithDistLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  Dtype* dist = top[0]->mutable_cpu_data();
  for (int i = 0; i < outer_num_; ++i) {
    const int dist_index = i*inner_num_;
    const int prob_index = i*dim;
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[dist_index + j]);
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      dist[dist_index + j] = -log(std::max(prob_data[prob_index + label_value * inner_num_ + j],
                           Dtype(FLT_MIN))) / inner_num_;
    }
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithDistLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* temp_data = temp_.mutable_cpu_data();
    int dim = prob_.count() / outer_num_;
    const int num_softmax = sum_multiplier_.count();

    caffe_copy(prob_.count(), prob_data, bottom_diff);
    for (int i = 0; i < outer_num_; ++i) {
      const int bottom_index = i * dim;
      const int label_index = i * inner_num_;
      // derivative of prob w.r.t. bottom
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[label_index + j]);
        bottom_diff[bottom_index + label_value * inner_num_ + j] -= 1;
      }
      // derivative of loss w.r.t. prob
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_softmax, inner_num_, 1,
          1. / inner_num_, sum_multiplier_.cpu_data(), top_diff + label_index,
          0., temp_data + bottom_index);
    }
    caffe_mul(temp_.count(), temp_data, bottom_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithDistLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithDistLayer);
REGISTER_LAYER_CLASS(SoftmaxWithDist);

}  // namespace caffe
