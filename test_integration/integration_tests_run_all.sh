#!/bin/bash

# Can be also empty, no test results will be delivered then.
result_output=$1

set -x

./integration-test-backgeocoding "$result_output"
res1=$?
./integration-test-coherence "$result_output"
res2=$?
./integration-test-snap-engine "$result_output"
res3=$?
./integration-test-tc "$result_output"
res4=$?
./integration-test-apply-orbit-file-op "$result_output"
res5=$?
./integration-sentinel1_product_reader "$result_output"
res6=$?
./integration-test-topsar-deburst-op "$result_output"
res7=$?
./integration-test-sentinel1-calibrate "$result_output"
res8=$?

exit_value=$((res1 | res2 | res3 | res4 | res5 | res6 | res7 | res8))
exit $exit_value