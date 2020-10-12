#!/bin/bash

# Can be also empty, no test results will be delivered then.
result_output=$1

set -x

./integration-test-backgeocoding $result_output
res1=$?
./integration-test-coherence $result_output
res2=$?
./integration-test-snap-engine $result_output
res3=$?
./integration-test-tc $result_output
res4=$?

exit_value=$(($res1 | $res2 | $res3 | $res4))
exit $exit_value
