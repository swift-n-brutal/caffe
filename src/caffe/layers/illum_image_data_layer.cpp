#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/illum_image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
IllumImageDataLayer<Dtype>::~IllumImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void IllumImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  IllumImageDataParameter illum_image_data_param = this->layer_param_.illum_image_data_param();
  const int new_height = illum_image_data_param.new_height();
  const int new_width  = illum_image_data_param.new_width();
  const int source_size = illum_image_data_param.source_size();
  const int mean_file_size = illum_image_data_param.mean_file_size();
  CHECK(mean_file_size == 0 || mean_file_size == source_size);
  const bool cache = illum_image_data_param.cache();
  string root_folder = illum_image_data_param.root_folder();
  const int crop_size = illum_image_data_param.crop_size();
  const int batch_size = illum_image_data_param.batch_size();
  const int label_dim = illum_image_data_param.label_dim();
  const int patch_per_image = illum_image_data_param.patch_per_image();
  CHECK(batch_size % patch_per_image == 0);

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  if (mean_file_size) {
	  for (int i = 0; i < source_size; ++i) {
		  const string& source = illum_image_data_param.source(i);
		  const string& mean_file = illum_image_data_param.mean_file(i);
		  LOG(INFO) << "Opening file " << source;
		  LOG(INFO) << "Opening file " << mean_file;
		  std::ifstream infile(source.c_str());
		  std::ifstream infile2(mean_file.c_str());
		  string filename, filename2;
		  float la, lb, lc;
		  while (infile >> filename >> la >> lb >> lc) {
			vector<float> vec3(3);
			vec3[0] = la;
			vec3[1] = lb;
                    vec3[2] = lc;
			infile2 >> filename2 >> la >> lb;
			CHECK(filename2 == filename);
			vec3[0] /= la;
			vec3[1] /= lb;
			lines_.push_back(std::make_pair(filename, vec3));
		  }
		  infile.close();
		  infile2.close();
	  }
  } else {
	  for (int i = 0; i < source_size; ++i) {
		  const string& source = illum_image_data_param.source(i);
		  LOG(INFO) << "Opening file " << source;
		  std::ifstream infile(source.c_str());
		  string filename;
		  float la, lb, lc;
		  while (infile >> filename >> la >> lb >> lc) {
			vector<float> vec3(2);
			vec3[0] = la;
			vec3[1] = lb;
                    vec3[2] = lc;
			lines_.push_back(std::make_pair(filename, vec3));
		  }
		  infile.close();
	  }
  }
  order_.resize(lines_.size());
  for (int ord = 0; ord < lines_.size(); ++ord) {
	order_.at(ord) = ord;
  }

  if (this->layer_param_.illum_image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";
  
  if (cache) {
	LOG(INFO) << "Read data to memory.";
	raw_data_.resize(lines_.size());
	for (int i = 0; i < lines_.size(); ++i) {
	  string image_path = root_folder + lines_.at(i).first + ".bin";
	  std::ifstream image_file(image_path.c_str(), std::ios::binary);
	  std::ostringstream osst;
	  osst << image_file.rdbuf();
	  raw_data_.at(i) = string(osst.str());
	  image_file.close();
	}
  }
	
  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.illum_image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.illum_image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  
  // Read an image, and use it to initialize the top blob.
  int hwc[3];
  string image_name = root_folder + lines_.at(order_.at(lines_id_)).first + ".bin";
  std::ifstream im_file(image_name.c_str(), std::ios::binary);
  im_file.read((char*)hwc, sizeof(int)*3);
  im_file.close();
  // image
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  if (crop_size > 0) {
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
	  this->prefetch_[i].data_.Reshape(batch_size, hwc[2], crop_size, crop_size);
    }
    top[0]->Reshape(batch_size, hwc[2], crop_size, crop_size);
  } else {
	LOG(ERROR) << "crop_size must be set.";
  }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  top[1]->Reshape(batch_size, label_dim, 1, 1);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(batch_size, label_dim, 1, 1);
  }
}

template <typename Dtype>
void IllumImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(order_.begin(), order_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void IllumImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  //CHECK(this->transformed_data_.count());
  IllumImageDataParameter illum_image_data_param = this->layer_param_.illum_image_data_param();
  const int batch_size = illum_image_data_param.batch_size();
  //const int new_height = illum_image_data_param.new_height();
  //const int new_width = illum_image_data_param.new_width();
  string root_folder = illum_image_data_param.root_folder();
  const bool cache = illum_image_data_param.cache();
  const int crop_size = illum_image_data_param.crop_size();
  const int label_dim = illum_image_data_param.label_dim();
  const int patch_per_image = illum_image_data_param.patch_per_image();
  int image_per_batch = batch_size / patch_per_image;
  const float black_thresh = illum_image_data_param.black_thresh();
  const float scale = illum_image_data_param.scale();
  const bool mirror = illum_image_data_param.mirror();
  int thres = 0;
  if (black_thresh >= 1) {
	thres = int(black_thresh);
  } else if (black_thresh > 0) {
	thres = int (crop_size * crop_size * black_thresh);
  }
  
  // Read an image, and use it to initialize the top blob.
  int hwc[3];
  if (cache) {
	int *c_ptr = (int *) &(raw_data_.at(0).at(sizeof(int)*2));
	hwc[2] = *c_ptr;
  } else {
	string image_name = root_folder + lines_.at(order_.at(lines_id_)).first + ".bin";
	std::ifstream im_file(image_name.c_str(), std::ios::binary);
	im_file.read((char*)hwc, sizeof(int)*3);
	im_file.close();
  }
  batch->data_.Reshape(batch_size, hwc[2], crop_size, crop_size);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  caffe::rng_t* prefetch_rng =
    static_cast<caffe::rng_t*>(prefetch_rng_->generator());

  // datum scales
  const int lines_size = lines_.size();
  int im_buffer_size = 0;
  float* im_buffer = NULL;
  int height, width, channels;
  int wh, size;
  for (int item_id = 0; item_id < image_per_batch; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
	int lines_id = order_.at(lines_id_);
	if (cache) {
	  int *hwc_ptr = (int *)&(raw_data_.at(lines_id).at(0));
	  height = hwc_ptr[0];
	  width = hwc_ptr[1];
	  channels = hwc_ptr[2];
	  size = height * width * channels;
	  wh = width * height;
	  im_buffer = (float*)&(raw_data_.at(lines_id).at(sizeof(int)*3));
	} else {
	  string image_bin_name = root_folder + lines_.at(lines_id).first + ".bin";
	  std::ifstream image_file(image_bin_name.c_str(), std::ios::binary);
	  image_file.read((char*)hwc, sizeof(int)*3);
	  height = hwc[0];
	  width = hwc[1];
	  channels = hwc[2];
	  size = height * width * channels;
	  wh = width * height;
	  if (im_buffer_size < size) {
            if (im_buffer_size) {delete im_buffer;}
		im_buffer = new float[size];
		im_buffer_size = size;
	  }
	  image_file.read((char*)im_buffer, sizeof(float)*size);
	  image_file.close();
	}
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
	for (int patch_id = 0; patch_id < patch_per_image; ++patch_id) {
	  int sample_id = item_id * patch_per_image + patch_id;
	  // image
	  if (crop_size) {
		int h_off, w_off;
		// ignore patches with too many black pixels
		int num_black = 0;
		do {
		  h_off = (*prefetch_rng)() % (height - crop_size);
		  w_off = (*prefetch_rng)() % (width - crop_size);
		  num_black = 0;
		  // TODO: count black pixels efficiently
		  for (int h = 0; h < crop_size && num_black <= thres; ++h) {
			for (int w = 0; w < crop_size && num_black <= thres; ++w) {
			  int data_index = (w_off + w) * height + h_off + h;
			  if (im_buffer[data_index] == 0 ||
				  im_buffer[data_index + wh] == 0) {
				++num_black;
			  } else if (channels >= 3 &&
				  im_buffer[data_index + wh + wh] == 0) {
				++num_black;
			  } else if (channels >= 4 &&
				  im_buffer[data_index + wh*3] == 0) {
				++num_black;
			  }
			}
		  }
		} while(num_black > thres);
		if (mirror && (*prefetch_rng)() % 2) {
		  // Copy mirrored version
		  for (int c = 0; c < channels; ++c) {
		    for (int h = 0; h < crop_size; ++h) {
			  for (int w = 0; w < crop_size; ++w) {
				int top_index = ((sample_id * channels + c) * crop_size + h)
					* crop_size + (crop_size - 1 - w);
				int data_index = (c * width + w_off + w) * height + h_off + h; // in matlab order
				Dtype datum_element =
					static_cast<Dtype>(im_buffer[data_index]);
				prefetch_data[top_index] = datum_element * scale;
			  }
			}
		  }
	    } else {
		  // Normal copy
		  for (int c = 0; c < channels; ++c) {
			for (int h = 0; h < crop_size; ++h) {
			  for (int w = 0; w < crop_size; ++w) {
				int top_index = ((sample_id * channels + c) * crop_size + h)
					* crop_size + w;
				int data_index = (c * width + w_off + w) * height + h_off + h; // in matlab order
				Dtype datum_element =
					static_cast<Dtype>(im_buffer[data_index]);
				prefetch_data[top_index] = datum_element * scale;
			  }
			}
		  }
	    }
	  } else {
		LOG(ERROR) << "crop_size must be set.";
	  }
	  // label
	  for (int i = 0; i < label_dim; ++i) {
		prefetch_label[sample_id * label_dim + i] = lines_.at(lines_id).second[i] * scale;
	  }
	}
    trans_time += timer.MicroSeconds();
	
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (illum_image_data_param.shuffle()) {
        ShuffleImages();
      }
    }
  }
  // clear temp buffer
  if (im_buffer_size) {
    delete im_buffer;
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(IllumImageDataLayer);
REGISTER_LAYER_CLASS(IllumImageData);

}  // namespace caffe
#endif  // USE_OPENCV
