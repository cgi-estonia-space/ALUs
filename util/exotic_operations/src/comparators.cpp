#include "comparators.hpp"

namespace slap {

int equalsArrays(float *a, float *b, int elems){
    int i,count =0;
    float temp;

    for(i=0; i<elems; i++){
        temp = (a[i]>b[i])*(a[i]-b[i]) + (a[i]<=b[i])*(b[i]-a[i]);
        if(temp > ERROR_RANGE){
            std::cerr << "elements do not match - " <<i<<")"<< a[i] << ":"<<b[i] << '\n';
            count++;
            if(count > 50){
                return count;
            }
        }
    }
    return count;
}

int equalsArraysd(double *a, double *b, int elems){
    int i,count =0;
    double temp;

    for(i=0; i<elems; i++){
        temp = (a[i]>b[i])*(a[i]-b[i]) + (a[i]<=b[i])*(b[i]-a[i]);
        if(temp > ERROR_RANGE){
            std::cerr << "elements do not match - " <<i<<")"<< a[i] << ":"<<b[i] << '\n';
            count++;
            if(count > 50){
                return count;
            }
        }
    }
    return count;
}

int equalsArrays2Dd(double **a, double **b, int x, int y){
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
