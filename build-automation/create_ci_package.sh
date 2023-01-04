#!/bin/bash

set -e

if [ "$#" -ne 2 ]; then
    echo "$0 requires repo location and output filename path."
    echo "Usage:"
    echo "$0 [ALUs repo] [package filename path]"
    exit 1
fi

cd $1
tar --exclude=docs --exclude-vcs --exclude-vcs-ignores -czvf $2 .
