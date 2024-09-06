#!/bin/bash

FMT_PATH="$(pwd)/../include/blade/utils/fmt"

find $FMT_PATH \( -name '*.h' -o -name '*.cc' \) -exec   sed -i 's/fmt::/bl::fmt::/g; s/namespace fmt/namespace bl::fmt/g; s/FMT_/BL_FMT_/g;' '{}' \;