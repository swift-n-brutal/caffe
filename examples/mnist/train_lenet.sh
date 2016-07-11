#!/usr/bin/env sh

GLOG_log_dir=examples/mnist/tmp/ ./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt
