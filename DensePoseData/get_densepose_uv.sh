#!/bin/bash
mkdir uv_data
cd uv_data
wget https://s3.amazonaws.com/densepose/densepose_uv_data.tar.gz
tar xvf densepose_uv_data.tar.gz
rm densepose_uv_data.tar.gz
