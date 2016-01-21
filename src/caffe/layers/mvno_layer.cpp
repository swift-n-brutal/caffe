#include <vector>

#include "caffe/layers/mvno_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MVNOLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  mean_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      1, 1);
  variance_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      1, 1);
  temp_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  vector<int> mv_shape(2);
  mv_shape[0] = bottom[0]->num();
  if ( this->layer_param_.mvno_param().across_channels() ) {
    sum_multiplier_.Reshape(1, bottom[0]->channels(), bottom[0]->height(),
                            bottom[0]->width());
    mv_shape[1] = 1;
  } else {
    sum_multiplier_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
    mv_shape[1] = bottom[0]->channels();
  }
  if (this->layer_param_.mvno_param().output_mean() ||
      this->layer_param_.mvno_param().output_variance()) {
    top[1]->Reshape(mv_shape);
    if (this->layer_param_.mvno_param().output_mean() &&
        this->layer_param_.mvno_param().output_variance()) {
      top[2]->Reshape(mv_shape);
    }
  }
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  eps_ = this->layer_param_.mvno_param().eps();
}

template <typename Dtype>
void MVNOLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
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
	caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
        sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  }
  
  if (this->layer_param_.mvno_param().sub_mean()) {
    variance_scale = 1. / dim;
    // subtract mean
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
        mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
        temp_.mutable_cpu_data());
    caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data); 
  }
  
  if (compute_variance) {
    // compute variance using var(X) = E((X-EX)^2) or length
    caffe_powx(bottom[0]->count(), top_data, Dtype(2),
        temp_.mutable_cpu_data());  // (X-EX)^2 or X^2
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, variance_scale, temp_.cpu_data(),
        sum_multiplier_.cpu_data(), 0.,
        variance_.mutable_cpu_data());  // E((X-EX)^2) or square of length
	caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
        variance_.mutable_cpu_data());  // sqrt(var) or sqrt(length^2)
  }
  
  if (this->layer_param_.mvno_param().normalize_variance()) {
    // normalize length of vector
    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
          temp_.mutable_cpu_data());

    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
  }
  
  if (this->layer_param_.mvno_param().output_mean()) {
    caffe_copy(num, mean_.cpu_data(), top[1]->mutable_cpu_data());
	if (this->layer_param_.mvno_param().output_variance()) {
      caffe_copy(num, variance_.cpu_data(), top[2]->mutable_cpu_data());
	}
  } else if (this->layer_param_.mvno_param().output_variance()) {
	caffe_copy(num, variance_.cpu_data(), top[1]->mutable_cpu_data());
  }
}

template <typename Dtype>
void MVNOLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  int num;
  if (this->layer_param_.mvno_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int dim = bottom[0]->count() / num;

  if (this->layer_param_.mvno_param().sub_mean()) {
    if (this->layer_param_.mvno_param().normalize_variance()) {
      caffe_mul(temp_.count(), top_data, top_diff, bottom_diff);
      caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., bottom_diff,
          sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
          bottom_diff);
      caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

      caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., top_diff,
            sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
            mean_.cpu_data(), sum_multiplier_.cpu_data(), 1.,
            bottom_diff);

      caffe_cpu_axpby(temp_.count(), Dtype(1), top_diff, Dtype(-1. / dim),
          bottom_diff);

      // put the squares of bottom into temp_ 
	  // TODO: delete the next line. it is not used.
      /*caffe_powx(temp_.count(), bottom_data, Dtype(2),
          temp_.mutable_cpu_data());*/
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
          temp_.mutable_cpu_data());

      caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);
    } else {
      caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, top_diff,
          sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
          mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
          temp_.mutable_cpu_data());
      caffe_add(temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
    }
  } else if (this->layer_param_.mvno_param().normalize_variance()){
	caffe_mul(temp_.count(), top_data, top_diff, bottom_diff);
	caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., bottom_diff,
	    sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
	    mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
		bottom_diff);
	caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);
	
	caffe_cpu_axpby(temp_.count(), Dtype(1.), top_diff, Dtype(-1.),
	    bottom_diff);
	
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
	    variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
		temp_.mutable_cpu_data());
	
	caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);
  } else {
	// do nothing
	caffe_copy(temp_.count(), top_diff, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(MVNOLayer);
#endif

INSTANTIATE_CLASS(MVNOLayer);
REGISTER_LAYER_CLASS(MVNO);

}  // namespace caffe
