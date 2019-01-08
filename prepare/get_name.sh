#!/bin/bash
while read p; do
  name=`mono /home/ubuntu/FastRDFStore/bin/FastRDFStoreClient.exe -m $p --pred type.object.name | tail -3 | head -1 | cut -f2 -d'>' | awk '{$1=$1;print}'`
  alias=`mono /home/ubuntu/FastRDFStore/bin/FastRDFStoreClient.exe -m $p --pred common.topic.alias | tail -3 | head -1 | cut -f2 -d'>' | awk '{$1=$1;print}'`

  if [[ $name =~ "Endpoint connected to" ]]; then
    name=""
  fi
  if [[ $alias =~ "Endpoint connected to" ]]; then
    alias=""
  fi

  echo -e "$p\t$name\t$alias" >> $2
done < $1