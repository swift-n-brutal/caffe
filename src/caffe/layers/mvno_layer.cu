#include <vector>

#include "caffe/layers/mvno_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MVNOLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const bool compute_mean = this->layer_param_.mvno_param().sub_mean() ||
      this->layer_param_.mvno_param().output_mean();
  const bool compute_variance = this->layer_param_.mvno_param().normalize_variance() ||
      this->layer_param_.mvno_param().output_variance();
  Dtype variance_scale = 1.;
  int num;
  if (this->layer_param_.mvno_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int dim = bottom[0]->count() / num;
  
  if (compute_mean) {
	caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
        sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());  // EX
  }
  
  if (this->layer_param_.mvno_param().sub_mean()) {
    variance_scale = 1. / dim;
    // subtract mean
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
        mean_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
        temp_.mutable_gpu_data());
    caffe_gpu_add(temp_.count(), bottom_data, temp_.gpu_data(), top_data);  // X-EX
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data); 
  }
  
  if (compute_variance) {
    // compute variance using var(X) = E((X-EX)^2) or length
    caffe_gpu_powx(bottom[0]->count(), top_data, Dtype(2),
        temp_.mutable_gpu_data());  // (X-EX)^2 or X^2
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, variance_scale, temp_.gpu_data(),
        sum_multiplier_.gpu_data(), 0.,
        variance_.mutable_gpu_data());  // E((X-EX)^2) or square of length
	caffe_gpu_powx(variance_.count(), variance_.gpu_data(), Dtype(0.5),
        variance_.mutable_gpu_data());  // sqrt(var) or sqrt(length^2)
  }
  
  if (this->layer_param_.mvno_param().normalize_variance()) {
    // normalize length of vector
    caffe_gpu_add_scalar(variance_.count(), eps_, variance_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
          temp_.mutable_gpu_data());

    caffe_gpu_div(temp_.count(), top_data, temp_.gpu_data(), top_data);
  }
  
  if (this->layer_param_.mvno_param().output_mean()) {
    caffe_copy(num, mean_.gpu_data(), top[1]->mutable_gpu_data());
	if (this->layer_param_.mvno_param().output_variance()) {
      caffe_copy(num, variance_.gpu_data(), top[2]->mutable_gpu_data());
	}
  } else if (this->layer_param_.mvno_param().output_variance()) {
	caffe_copy(num, variance_.gpu_data(), top[1]->mutable_gpu_data());
  }
}

template <typename Dtype>
void MVNOLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  int num;
  if (this->layer_param_.mvno_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int dim = bottom[0]->count() / num;

  if (this->layer_param_.mvno_param().sub_mean()) {
    if (this->layer_param_.mvno_param().normalize_variance()) {
      caffe_gpu_mul(temp_.count(), top_data, top_diff, bottom_diff);
      caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., bottom_diff,
          sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          mean_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
          bottom_diff);
      caffe_gpu_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

      caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., top_diff,
            sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
            mean_.gpu_data(), sum_multiplier_.gpu_data(), 1.,
            bottom_diff);

      caffe_gpu_axpby(temp_.count(), Dtype(1), top_diff, Dtype(-1. / dim),
          bottom_diff);

      // put the squares of bottom into temp_ 
	  // TODO: delete the next line. it is not used.
      /*caffe_gpu_powx(temp_.count(), bottom_data, Dtype(2),
          temp_.mutable_gpu_data());*/
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
          temp_.mutable_gpu_data());

      caffe_gpu_div(temp_.count(), bottom_diff, temp_.gpu_data(), bottom_diff);
    } else {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, top_diff,
          sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
          mean_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
          temp_.mutable_gpu_data());
      caffe_gpu_add(temp_.count(), top_diff, temp_.gpu_data(), bottom_diff);
    }
  } else if (this->layer_param_.mvno_param().normalize_variance()){
	caffe_gpu_mul(temp_.count(), top_data, top_diff, bottom_diff);
	caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., bottom_diff,
	    sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
	    mean_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
		bottom_diff);
	caffe_gpu_mul(temp_.count(), top_data, bottom_diff, bottom_diff);
	
	caffe_gpu_axpby(temp_.count(), Dtype(1.), top_diff, Dtype(-1.),
	    bottom_diff);
	
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
	    variance_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
		temp_.mutable_gpu_data());
	
	caffe_gpu_div(temp_.count(), bottom_diff, temp_.gpu_data(), bottom_diff);
  } else {
	// do nothing
	caffe_copy(temp_.count(), top_diff, bottom_diff);
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(MVNOLayer);


}  // namespace caffe
