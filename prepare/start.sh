#!/bin/bash

base_dir=$(cd $1; pwd)
out_dir=$(cd $2; pwd)
for file_name in $(ls $base_dir); do
    _in=$base_dir"/"$file_name
    _out=$out_dir"/"out_$file_name
    nohup ./getname.sh $_in $_out &
done