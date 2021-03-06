#!/bin/bash

set -e

if [ -z "$PREFIX" ]; then
  PREFIX="$CONDA_PREFIX"
fi

# Point CMake to use conda specific paths.
export CMAKE_LIBRARY_PATH=$PREFIX/lib:$PREFIX/include:$CMAKE_LIBRARY_PATH

# conda build will copy everything over, including build directories.
# Don't let this pollute hte build!
rm -rf build || true

PYTHON_ARGS="$(python ./scripts/get_python_cmake_flags.py)"

mkdir -p build
cd build
# TODO(Yangqing): turn on rocksdb when available.
cmake \
    -DBLAS=MKL \
    -DMKL_INCLUDE_DIR=$CONDA_PREFIX/include \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_PREFIX_PATH="$PREFIX" \
    -DUSE_ROCKSDB=OFF \
    $CONDA_CMAKE_ARGS \
    $PYTHON_ARGS \
    ..
# which cmake && exit 1
make -j$(nproc)

make install/fast

# Python libraries got installed to wrong place, so move them
# to the right place. See https://github.com/caffe2/caffe2/issues/1015
echo "Installing Python to $SP_DIR"
mv $PREFIX/caffe2 $SP_DIR
