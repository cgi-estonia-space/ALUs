#!/bin/bash

function is_error_then_quit {
	ret_val=$?
	if [ $ret_val -ne 0 ]; then
		echo "Exiting because of error no $ret_val"
		exit $ret_val
	fi
}

# This script runs the CI routine for the project.
rm -rf build
cmake . -Bbuild -DENABLE_TESTS=ON
is_error_then_quit
cd build
is_error_then_quit
make -j8
is_error_then_quit
cd unit-test
is_error_then_quit
cp ../../build-automation/execute_execs.sh .
is_error_then_quit
bash execute_execs.sh --gtest_output=xml:test-results/
unit_test_success=$?
cd ../test_integration
is_error_then_quit
cp ../../test_integration/integration_tests_run_all.sh .
is_error_then_quit
./integration_tests_run_all.sh --gtest_output=xml:test-results/
integration_test_success=$?

exit $(($unit_test_success | $integration_test_success))

