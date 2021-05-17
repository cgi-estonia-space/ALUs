# Coding style

In the root of the project is conforming [clang-format](.clang-format) and [clang-tidy](.clang-tidy) files that should be used formatting the code.\
Code style is following [Google C++ style guide](https://google.github.io/styleguide/cppguide.html) with some exceptions.

## Exceptions to Google C++ style guide

### Placement of reference '&' and pointer "*"

https://google.github.io/styleguide/cppguide.html#Pointer_and_Reference_Expressions

One should do this consistently within a single file, so, when modifying an existing file, use the style in that file.

```
// These are fine, space preceding.
char *c;
const std::string &str;

// These are fine, space following.
char* c;
const std::string& str;
```

Putting the const first is arguably more readable, since it follows English in putting the "adjective" (const) before the "noun" (int).\
Stylesheet recommends this, but does not require it (it asks to be consistent), but will use form below (const left of type):

```
const int foo
```

### Header files
Cuda files use extension .cu and .cuh for implementation and declaration respectively

### The #define guard
Use #pragma once

### Forward declarations
Exceptions are all CUDA files where forward declaration is often needed.

### Names and order of includes
Keep C headers and C++ headers in the same block/group.
This is because of the fact that C headers have been merged into modern C++ already which means that we always include c headers with a name prefix "c".\

For example instead of\
``#include <stdio.h>``\
use\
``#include <cstdio>``

### Unnamed Namespaces and Static Variables
Use unnamed namespaces to declare "static" variables.

### Constant names
Use only uppercase characters with underscore in between.\
ALSO! Same applies for class encapsulated constants, therefore no trailing underscore.\
Exceptions are function arguments and local variables which follow 'lower_case'.

### Exceptions

Using exceptions for this project is encouraged.

### Comments
Doxygen comment approach is used for files, functions, classes and variables.
More sophisticated are function comments like:

```
/**
* Compute zero Doppler time for given earth point using bisection method.
*
* Duplicate of a SNAP's SARGeocoding.java's getEarthPointZeroDopplerTime().
*
* @param firstLineUTC The zero Doppler time for the first range line.
* @param lineTimeInterval The line time interval.
* @param wavelength The radar wavelength.
* @param earthPoint The earth point in xyz coordinate.
* @param sensorPosition Array of sensor positions for all range lines.
* @param sensorVelocity Array of sensor velocities for all range lines.
* @return The zero Doppler time in days if it is found, -1 otherwise.
*/
``` 
  
### Line Length
Line length is 120 characters.

### Spaces vs. Tabs
4 spaces to indent.

# Development build verification

For verification of code base a script can be called from the root of the repo

```
build-automation/build_and_run_ci.sh
```

Same script is run by the CI verification. This runs all the unit and integration tests too.
