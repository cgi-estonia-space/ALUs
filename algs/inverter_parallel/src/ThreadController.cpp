#include "ThreadController.hpp"

ThreadController::ThreadController(AlgoData data){
    this->threadCount = 8;
    this->workerCount = 0;
    this->finishedCount = 0;
    this->algoData = data;
}

void ThreadController::startThreads(){
    int i=0;
    
    for(algoData.tileXo= 0; algoData.tileXo <algoData.rasterX; algoData.tileXo+= algoData.tileX){
        for(algoData.tileYo = 0; algoData.tileYo < algoData.rasterY; algoData.tileYo+= algoData.tileY){
            this->computeTileActual(algoData.rasterX, algoData.rasterY, algoData.tileX, algoData.tileY, algoData.tileXo, algoData.tileYo, &algoData.tileXa, &algoData.tileYa);
            new ThreadHolder(algoData, this);
            this->workerCount++;
        }
    }
    for(i=0; i<this->threadCount; i++){
        this->threadSync.notify_one();
    }

    std::mutex finalMutex;
    std::unique_lock<std::mutex> lk(finalMutex);
    this->endBlock.wait(lk);
    std::cout<<"final block reached"<< std::endl;
}

void ThreadController::computeTileActual(int rasterX, int rasterY, int tileX, int tileY, int tileXo, int tileYo, int *tileXa, int *tileYa){
    *tileXa = rasterX - tileXo;
    *tileYa = rasterY - tileYo;

    *tileXa = (*tileXa > tileX)*tileX + !(*tileXa > tileX)* *tileXa;
    *tileYa = (*tileYa > tileY)*tileY + !(*tileYa > tileY)* *tileYa;
}

void ThreadController::registerThreadEnd(CPLErr error, AlgoData data){
    this->registerMutex.lock();

    this->finishedCount++;
    if(error){
        this->erroredConfs.push_back(data);
    }
    if(this->finishedCount == this->workerCount){
        this->endBlock.notify_all();
    }else{
        this->threadSync.notify_one();
    }
    this->registerMutex.unlock();
}
