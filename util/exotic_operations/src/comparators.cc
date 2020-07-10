#include "comparators.h"

namespace alus {

int EqualsArrays(const float *a, const float *b, int elems, float delta){
    int i,count =0;
    float temp;

    for(i=0; i<elems; i++){
        temp = (a[i]>b[i])*(a[i]-b[i]) + (a[i]<=b[i])*(b[i]-a[i]);
        if(temp > delta){
            std::cerr << "elements do not match - " <<i<<")"<< a[i] << ":"<<b[i] << '\n';
            count++;
            if(count > 50){
                return count;
            }
        }
    }
    return count;
}

int EqualsArrays(const float *a,const float *b, int elems){
    return EqualsArrays(a, b, elems, ERROR_RANGE);
}



int EqualsArraysd(const double *a, const double *b, int elems){
    return EqualsArraysd(a, b, elems, ERROR_RANGE);
}

int EqualsArraysd(const double *a, const double *b, int elems, double delta){
    int i,count =0;
    double temp;

    for(i=0; i<elems; i++){
        temp = (a[i]>b[i])*(a[i]-b[i]) + (a[i]<=b[i])*(b[i]-a[i]);
        if(temp > delta){
            std::cerr << "elements do not match - " <<i<<")"<< a[i] << ":"<<b[i] << '\n';
            count++;
            if(count > 50){
                return count;
            }
        }
    }
    return count;
}

int EqualsArrays2Dd(const double *const *a, const double *const *b, int x, int y){
    int i,j, count=0;
    double temp;

    for(i=0; i<x; i++){
        for(j=0; j<y; j++){
            temp = (a[i][j]>b[i][j])*(a[i][j]-b[i][j]) + (a[i][j]<=b[i][j])*(b[i][j]-a[i][j]);
            if(temp > ERROR_RANGE){
                std::cerr << "elements do not match - " <<i<<","<<j<<")"<< a[i][j] << ":"<<b[i][j] << '\n';
                count++;
                if(count > 50){
                    return count;
                }
            }

        }
    }
    return count;
}

}//namespace
