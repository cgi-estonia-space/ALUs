#pragma once

namespace slap {

class UTC{
private:

public:
    int days, seconds, microseconds;

    UTC(){

    }

    UTC(int days, int seconds, int microseconds){
        this->days = days;
        this->seconds = seconds;
        this->microseconds = microseconds;
    }

};

}//namespace
