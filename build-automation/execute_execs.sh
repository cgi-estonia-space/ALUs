#!/bin/bash

test_results=$1

result=0

while read -r exe
do
    ./$exe $test_results
    last_result=$?
    result=$(($result | $last_result))
done < <(find . -maxdepth 1 -executable -type f)

exit $result
