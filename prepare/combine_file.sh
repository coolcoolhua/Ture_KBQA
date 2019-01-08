#!/bin/bash

base_dir=$(cd $1; pwd)
out_dir=$(cd $2; pwd)
file_out='fb_data_names.txt'
for file_name in $(ls $base_dir); do
    _in=$base_dir"/"$file_name
    cat $_in >> $out_dir"/"$file_out
done