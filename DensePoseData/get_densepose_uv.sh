#!/bin/bash
mkdir UV_data
cd UV_data
wget https://s3.amazonaws.com/densepose/densepose_uv_data.tar.gz
tar xvf densepose_uv_data.tar.gz
rm densepose_uv_data.tar.gz
