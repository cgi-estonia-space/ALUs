#pragma once
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <condition_variable>
#include "AlgoData.hpp"

class ThreadController{
private:
    int threadCount;
    int workerCount;
    int finishedCount;
    std::condition_variable threadSync;
    std::condition_variable endBlock;
    std::mutex registerMutex;
    std::mutex outMutex;
    AlgoData algoData;
    std::vector<AlgoData> erroredConfs;

    void computeTileActual(int rasterX, int rasterY, int tileX, int tileY, int tileXo, int tileYo, int *tileXa, int *tileYa);
public:
    ThreadController(AlgoData data);
    void startThreads();
    void registerThreadEnd(CPLErr error, AlgoData data);

    std::condition_variable * getThreadSync(){
        return &threadSync;
    }

    std::mutex *getOutMutex(){
        return &outMutex;
    }
    std::vector<AlgoData> *getErroredConfs(){
        return &this->erroredConfs;
    }
};


class ThreadHolder{
private:
    AlgoData algoData;
    std::thread *worker;
    ThreadController *controller;
    std::mutex mutex1;
public:
    ThreadHolder(AlgoData algoData, ThreadController *controller);
    void work();
    CPLErr invertColors(AlgoData data);
};
