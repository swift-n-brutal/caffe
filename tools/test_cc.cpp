#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;
using std::vector;
namespace db = caffe::db;

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int test_cc(int argc, char** argv);

int main(int argc, char** argv) {
//  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
  return test_cc(argc, argv);
}

// template<typename Dtype>
int test_cc(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  const int number_required_args_min = 6;
  const int number_required_args_max = number_required_args_min + 3;
  if (argc < number_required_args_min || argc > number_required_args_max) {
    LOG(ERROR) << "test_cc net_proto_file pretrained_net_proto_file"
    " image_folder/ image_names result_folder/"
    " [CPU/GPU] [DEVICE ID] [mean_file]";
  }
  
  if (argc > number_required_args_min && strcmp(argv[number_required_args_min], "GPU") == 0) {
    LOG(ERROR) << "Using GPU";
    int device_id = 0;
    if (argc > number_required_args_min + 1) {
      device_id = atoi(argv[number_required_args_min + 1]);
      CHECK_GE(device_id, 0);
    }
    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }
  
  std::string test_cc_proto(argv[1]);
  boost::shared_ptr<Net<float> > test_cc_net(
      new Net<float>(test_cc_proto, caffe::TEST));
  std::string pretrained_proto(argv[2]);
  LOG(INFO) << "Initializing net from " << pretrained_proto;
  test_cc_net->CopyTrainedLayersFrom(pretrained_proto);
  
  LOG(ERROR) << "Reading image names in " << argv[4];
  std::ifstream image_names_file(argv[4]);
  std::string root_folder(argv[3]);
  vector<std::string> image_names;
  std::string result_dir(argv[5]);
  std::string image_name;
  float lr, lg, lb;
  while (image_names_file >> image_name >> lr >> lb >> lg) {
    image_names.push_back(image_name);
  }
  image_names_file.close();
  LOG(ERROR) << "A total of " << image_names.size() << " images.";
  
  // read mean file if has
  bool add_mean = false;
  vector<std::pair<float, float> > scale_vec;
  float scale_r = 1.0, scale_b = 1.0;
  if (argc == number_required_args_max) {
    add_mean = true;
    LOG(ERROR) << "Reading mean file " << argv[8];
    std::ifstream mean_file(argv[8]);
    while (mean_file >> image_name >> scale_r >> scale_b) {
      scale_vec.push_back(std::make_pair(scale_r, scale_b));
    }
    mean_file.close();
    CHECK_EQ(scale_vec.size(), image_names.size());
  }
  
  const float scale = 1.0;
  scale_r = 1.0;
  scale_b = 1.0;
  const int batch_size = test_cc_net->input_blobs()[0]->num();
  const int crop_size = test_cc_net->input_blobs()[0]->height();
  const int step = 10;
  int hwc[3];
  int height, width, channels = test_cc_net->input_blobs()[0]->channels();
  int wh;
  int size = 0;
  float* im_buffer = NULL;
  
  LOG(ERROR) << "Writing results to " << result_dir;
  float* input_data = test_cc_net->input_blobs()[0]->mutable_cpu_data();
  for (int i = 0; i < image_names.size(); ++i) {
    // read one image
    image_name = root_folder + image_names[i] + ".bin";
    string result_name = result_dir + image_names[i] + ".lumap";
    LOG(ERROR) << "(" << i + 1 << "/" << image_names.size() << ")" << image_name;
    std::ifstream image_file(image_name.c_str(), std::ios::binary);
    std::ofstream result_file(result_name.c_str());
    image_file.read((char*)hwc, sizeof(int)*3);
    height = hwc[0];
    width = hwc[1];
    channels = hwc[2];
    wh = width * height;
    if (size < wh * channels) {
      if (size > 0) { delete im_buffer; }
      size = wh * channels;
      im_buffer = new float[size];
    }
    image_file.read((char*)im_buffer, sizeof(float) * wh * channels);
    image_file.close();
    if (add_mean) {
      scale_r = scale_vec[i].first;
      scale_b = scale_vec[i].second;
    }
    // offsets for patches
    vector<int> h_offs, w_offs;
    for (int h_off = 0; h_off <= height - crop_size; h_off += step) {
      h_offs.push_back(h_off);
    }
    if ((height - crop_size) % step) {
      h_offs.push_back(height - crop_size);
    }
    for (int w_off = 0; w_off <= width - crop_size; w_off += step) {
      w_offs.push_back(w_off);
    }
    if ((width - crop_size) % step) {
      w_offs.push_back(width - crop_size);
    }
    int n_h_offs = h_offs.size();
    int n_w_offs = w_offs.size();
    result_file << n_h_offs << " " << n_w_offs << " " << step << " " << crop_size << "\n";
    int sample_id = 0;
    vector<std::pair<int, int> > pos;
    pos.resize(batch_size);
    for (int idh = 0; idh < n_h_offs; ++idh) {
      for (int idw = 0; idw < n_w_offs; ++idw) {
        int h_off = h_offs[idh], w_off = w_offs[idw];
        bool no_zero = true;
        for (int c = 0; c < channels && no_zero; ++c) {
          for (int h = 0; h < crop_size && no_zero; ++h) {
            for (int w = 0; w < crop_size && no_zero; ++w) {
              int data_index = (c * width + w_off + w) * height + h_off + h;
              int top_index = ((sample_id * channels + c) * crop_size + h)
                  * crop_size + w;
              if (im_buffer[data_index] == 0) {
                no_zero = false;
              }
              input_data[top_index] = im_buffer[data_index] * scale;
            }
          }
        } // end crop patch starting at (h_offs[idh], w_offs[idw])
        if (!no_zero) { continue; }
        pos[sample_id++] = std::make_pair(h_off, w_off);
        if (sample_id == batch_size) {
          if (Caffe::mode() == Caffe::GPU) {
            CUDA_CHECK(cudaMemcpy(test_cc_net->input_blobs()[0]->mutable_gpu_data(),
                input_data, sizeof(float) * test_cc_net->input_blobs()[0]->count(),
                cudaMemcpyHostToDevice));
          }
          const vector<Blob<float>*>& result = test_cc_net->ForwardPrefilled();
          const int label_dim = result[0]->channels();
          const float* pred = result[0]->cpu_data();
          if (label_dim == 3) {
            for (int k = 0; k < batch_size; ++k) {
              int kstart = k * label_dim;
              result_file << pos[k].first << " " << pos[k].second
                  << " " << pred[kstart] << " " << pred[kstart + 1]
                  << " " << pred[kstart + 2] << "\n";
            }
          } else if (label_dim == 2){
            for (int k = 0; k < batch_size; ++k) {
              int kstart = k * label_dim;
              result_file << pos[k].first << " " << pos[k].second
                  << " " << pred[kstart] * scale_r << " " << 1.0
                  << " " << pred[kstart + 1] * scale_b << "\n";
            }
          }
          sample_id = 0;
        }
      }
    } // end crop the whole image
    if (sample_id != 0) {
      // process the remaining patches
      if (Caffe::mode() == Caffe::GPU) {
        CUDA_CHECK(cudaMemcpy(test_cc_net->input_blobs()[0]->mutable_gpu_data(),
            input_data, sizeof(float) * test_cc_net->input_blobs()[0]->count(),
            cudaMemcpyHostToDevice));
      }
      const vector<Blob<float>*>& result = test_cc_net->ForwardPrefilled();
      const int label_dim = result[0]->channels();
      const float* pred = result[0]->cpu_data();
      if (label_dim == 3) {
        for (int k = 0; k < batch_size; ++k) {
          int kstart = k * label_dim;
          result_file << pos[k].first << " " << pos[k].second
              << " " << pred[kstart] << " " << pred[kstart + 1]
              << " " << pred[kstart + 2] << "\n";
        }
      } else if (label_dim == 2){
        for (int k = 0; k < batch_size; ++k) {
          int kstart = k * label_dim;
          result_file << pos[k].first << " " << pos[k].second
              << " " << pred[kstart] * scale_r << " " << 1.0
              << " " << pred[kstart + 1] * scale_b << "\n";
        }
      }
    } // end process the remaining patches
    result_file.close();
  } // end predict the illuminants of all images
  if (size > 0) { delete im_buffer; }
  return 0;
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  const int num_required_args = 7;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a trained network and an input data layer, and then"
    " extract features of the input data produced by the net.\n"
    "Usage: extract_features  pretrained_net_param"
    "  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]"
    "  save_feature_dataset_name1[,name2,...]  num_mini_batches  db_type"
    "  [CPU/GPU] [DEVICE_ID=0]\n"
    "Note: you can extract multiple features in one pass by specifying"
    " multiple feature blob names and dataset names separated by ','."
    " The names cannot contain white space characters and the number of blobs"
    " and datasets must be equal.";
    return 1;
  }
  int arg_pos = num_required_args;

  arg_pos = num_required_args;
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
    LOG(ERROR)<< "Using GPU";
    int device_id = 0;
    if (argc > arg_pos + 1) {
      device_id = atoi(argv[arg_pos + 1]);
      CHECK_GE(device_id, 0);
    }
    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  arg_pos = 0;  // the name of the executable
  std::string pretrained_binary_proto(argv[++arg_pos]);

  // Expected prototxt contains at least one data layer such as
  //  the layer data_layer_name and one feature blob such as the
  //  fc7 top blob to extract features.
  /*
   layers {
     name: "data_layer_name"
     type: DATA
     data_param {
       source: "/path/to/your/images/to/extract/feature/images_leveldb"
       mean_file: "/path/to/your/image_mean.binaryproto"
       batch_size: 128
       crop_size: 227
       mirror: false
     }
     top: "data_blob_name"
     top: "label_blob_name"
   }
   layers {
     name: "drop7"
     type: DROPOUT
     dropout_param {
       dropout_ratio: 0.5
     }
     bottom: "fc7"
     top: "fc7"
   }
   */
  std::string feature_extraction_proto(argv[++arg_pos]);
  boost::shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto, caffe::TEST));
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

  std::string extract_feature_blob_names(argv[++arg_pos]);
  std::vector<std::string> blob_names;
  boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));

  std::string save_feature_dataset_names(argv[++arg_pos]);
  std::vector<std::string> dataset_names;
  boost::split(dataset_names, save_feature_dataset_names,
               boost::is_any_of(","));
  CHECK_EQ(blob_names.size(), dataset_names.size()) <<
      " the number of blob names and dataset names must be equal";
  size_t num_features = blob_names.size();

  for (size_t i = 0; i < num_features; i++) {
    CHECK(feature_extraction_net->has_blob(blob_names[i]))
        << "Unknown feature blob name " << blob_names[i]
        << " in the network " << feature_extraction_proto;
  }

  int num_mini_batches = atoi(argv[++arg_pos]);

  std::vector<boost::shared_ptr<db::DB> > feature_dbs;
  std::vector<boost::shared_ptr<db::Transaction> > txns;
  const char* db_type = argv[++arg_pos];
  for (size_t i = 0; i < num_features; ++i) {
    LOG(INFO)<< "Opening dataset " << dataset_names[i];
    boost::shared_ptr<db::DB> db(db::GetDB(db_type));
    db->Open(dataset_names.at(i), db::NEW);
    feature_dbs.push_back(db);
    boost::shared_ptr<db::Transaction> txn(db->NewTransaction());
    txns.push_back(txn);
  }

  LOG(ERROR)<< "Extacting Features";

  Datum datum;
  std::vector<Blob<float>*> input_vec;
  std::vector<int> image_indices(num_features, 0);
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    feature_extraction_net->Forward(input_vec);
    for (int i = 0; i < num_features; ++i) {
      const boost::shared_ptr<Blob<Dtype> > feature_blob =
        feature_extraction_net->blob_by_name(blob_names[i]);
      int batch_size = feature_blob->num();
      int dim_features = feature_blob->count() / batch_size;
      const Dtype* feature_blob_data;
      for (int n = 0; n < batch_size; ++n) {
        datum.set_height(feature_blob->height());
        datum.set_width(feature_blob->width());
        datum.set_channels(feature_blob->channels());
        datum.clear_data();
        datum.clear_float_data();
        feature_blob_data = feature_blob->cpu_data() +
            feature_blob->offset(n);
        for (int d = 0; d < dim_features; ++d) {
          datum.add_float_data(feature_blob_data[d]);
        }
        string key_str = caffe::format_int(image_indices[i], 10);

        string out;
        CHECK(datum.SerializeToString(&out));
        txns.at(i)->Put(key_str, out);
        ++image_indices[i];
        if (image_indices[i] % 1000 == 0) {
          txns.at(i)->Commit();
          txns.at(i).reset(feature_dbs.at(i)->NewTransaction());
          LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
              " query images for feature blob " << blob_names[i];
        }
      }  // for (int n = 0; n < batch_size; ++n)
    }  // for (int i = 0; i < num_features; ++i)
  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
  // write the last batch
  for (int i = 0; i < num_features; ++i) {
    if (image_indices[i] % 1000 != 0) {
      txns.at(i)->Commit();
    }
    LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
        " query images for feature blob " << blob_names[i];
    feature_dbs.at(i)->Close();
  }

  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}
