#!/bin/bash
echo "IN: "$1
echo "OUT: "$2
echo "" > $2
OLD_IFS=$IFS
IFS=$'\n'
cat $1 | while read p; do
	echo "SPARQL PREFIX : <http://rdf.freebase.com/ns/> SELECT * FROM <http://freebase.com> WHERE {  :$p :type.object.name ?o };" > tmp
	for name in $(isql-vt localhost:1111 dba dba tmp | sed '1,9d;N;$d;P;D'); do
		echo -e "fb:$p\tfb:type.object.name\t$name" >> $2
	done
	echo "SPARQL PREFIX : <http://rdf.freebase.com/ns/> SELECT * FROM <http://freebase.com> WHERE {  :$p :common.topic.alias ?o };" > tmp
	for alias in $(isql-vt localhost:1111 dba dba tmp | sed '1,9d;N;$d;P;D'); do
		echo -e "fb:$p\tfb:common.topic.alias\t$alias" >> $2
	done
done
rm tmp
IFS=$OLD_IFS