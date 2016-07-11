// This is the hashing image data layer for single label images, and 
// the labels are category labels. The category labels should start with 0,
// and increase at the step of 1.
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/hashing_image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
HashingImageDataLayer<Dtype>::~HashingImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void HashingImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  vector<int> label;
  int temp;
  while (infile >> filename) {
    infile >> temp;
    label.push_back(temp);
	category.push_back(temp);
    lines_.push_back(std::make_pair(filename, label));
    label.clear();
  }

  // get the number of categories
  sort(category.begin(), category.end());
  vector<int>::iterator new_end;
  new_end = unique(category.begin(), category.end());
  category.erase(new_end, category.end());

  vector<int> temp_vec;
  for (int i = 0; i < category.size(); i++){
	  idx.push_back(temp_vec);
  }

  for (int i = 0; i < lines_.size(); i++) {
	  idx[lines_[i].second[0]].push_back(i);
  }

  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));

/*  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
	ShuffleImages();
  }*/
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  //vector<int> label_shape(batch_size, datum.label_size(), 1, 1);
  //top[1]->Reshape(label_shape);
  //vector<int> label_shape(1, this->layer_param_.data_param().batch_size());
  top[1]->Reshape(batch_size, 1, 1, 1);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    //this->prefetch_[i].label_.Reshape(label_shape);
    this->prefetch_[i].label_.Reshape(batch_size, 1, 1, 1);
  }
}

// This function is called on prefetch thread
template <typename Dtype>
void HashingImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();
  const int cat_per_iter = this->layer_param_.image_data_param().cat_per_iter();
  int item_per_cat = batch_size/cat_per_iter;
  CHECK_EQ(cat_per_iter * item_per_cat, batch_size) << "The batch size must be divisible by the cat_per_iter parameter.";

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(category.begin(), category.end(), prefetch_rng);
  for (int cat_id = 0; cat_id < cat_per_iter; cat_id++) {
	  shuffle(idx[category[cat_id]].begin(), idx[category[cat_id]].end(), prefetch_rng);
	  for (int item_id = 0; item_id < item_per_cat; ++item_id) {
		  // std::cout << lines_[idx[category[cat_id]][item_id]].first << lines_[idx[category[cat_id]][item_id]].second[0] << std::endl;
		  // get a blob
		  timer.Start();
		  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[idx[category[cat_id]][item_id]].first,
			  new_height, new_width, is_color);
		  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
		  read_time += timer.MicroSeconds();
		  timer.Start();
		  // Apply transformations (mirror, crop...) to the image
		  int offset = batch->data_.offset( cat_id * item_per_cat + item_id);
		  this->transformed_data_.set_cpu_data(prefetch_data + offset);
		  this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
		  trans_time += timer.MicroSeconds();
		  prefetch_label[cat_id * item_per_cat + item_id] = lines_[idx[category[cat_id]][item_id]].second[0];
	  }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(HashingImageDataLayer);
REGISTER_LAYER_CLASS(HashingImageData);

}  // namespace caffe
#endif  // USE_OPENCV
