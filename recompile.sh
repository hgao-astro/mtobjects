#!/bin/bash

libdirectory="${0%/*}/mto/clib"

if [ ! -d $libdirectory ];
then mkdir $libdirectory
fi

cd "${0%/*}/mto/clib"

socount=`ls *.so 2>/dev/null | wc -l`
if [ $socount -gt 0 ]
then rm *.so
fi


cd "${0%/*}/../csrc"

gcc -shared -fPIC -include main.h -o ../clib/mt_objects.so mt_objects.c mt_heap.c mt_node_test_4.c
gcc -shared -fPIC -include main.h -o ../clib/maxtree.so maxtree.c mt_stack.c mt_heap.c

gcc -shared -fPIC -include main_double.h  -o ../clib/mt_objects_double.so mt_objects.c mt_heap.c mt_node_test_4.c
gcc -shared -fPIC -include main_double.h -o ../clib/maxtree_double.so maxtree.c mt_stack.c mt_heap.c
