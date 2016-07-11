#!/usr/bin/env sh

GLOG_log_dir=examples/imagenet/tmp_0/ GLOG_logtostderr=0 ./build/tools/caffe train \
    --solver=models/bvlc_alexnet/solver.prototxt
