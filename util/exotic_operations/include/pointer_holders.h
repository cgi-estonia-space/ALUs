#pragma once

namespace alus{


/**
    The whole point of this is to store a matrix or a cube of data on the gpu.
    The problem is that in some cases, we can have several of these data piles and they can have
    different sizes and meanings. This is also a reason why there is an ID field here, as it may
    be necessary to tell different data piles apart, that are entered in a random order.
*/
struct PointerHolder{
    int id;
    void *pointer;
    int x;
    int y;
    int z;
};

struct PointerArray{
    PointerHolder *array;
};

}//namespace
