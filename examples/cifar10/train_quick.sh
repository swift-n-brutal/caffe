#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_log_dir=examples/cifar10/tmp/ GLOG_logtostderr=0 $TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver.prototxt

# reduce learning rate by factor of 10 after 8 epochs
GLOG_log_dir=examples/cifar10/tmp/ GLOG_logtostderr=0 $TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver_lr1.prototxt \
  --snapshot=examples/cifar10/cifar10_quick_iter_90000.solverstate.h5
