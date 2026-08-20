#!/bin/bash

#// Source - https://stackoverflow.com/a/36046965
#// Posted by Antimony, modified by community. See post 'Timeline' for change history
#// Retrieved 2026-03-18, License - CC BY-SA 4.0

#find foo/bar/ -iname '*.h' -o -iname '*.cpp' | xargs clang-format -i

find include/ -iname '*.h' -o -iname '*.cpp' -o -iname '*.cu' -o -iname '*.cuh' | xargs clang-format-22 -i
find test/ -iname '*.h' -o -iname '*.cpp' -o -iname '*.cu' -o -iname '*.cuh' | xargs clang-format-22 -i
