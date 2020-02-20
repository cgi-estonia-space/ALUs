#include "ThreadController.hpp"

ThreadHolder::ThreadHolder(AlgoData algoData, ThreadController *controller){
    this->algoData = algoData;
    this->controller = controller;
    std::thread worker(&ThreadHolder::work, this);
    worker.detach();
}

void ThreadHolder::work(){
    std::unique_lock<std::mutex> lk(this->mutex1);
    this->controller->getThreadSync()->wait(lk);

    CPLErr error = this->invertColors(this->algoData);


    if(error){
      this->controller->getOutMutex()->lock();
        std::cerr<<"There was error" << error << ". This incident will be retried. " << std::endl;
        std::cerr<< algoData.tileXo << ":" <<algoData.tileYo <<std::endl;
        this->controller->getOutMutex()->unlock();
    }

    this->controller->registerThreadEnd(error, this->algoData);
}

CPLErr ThreadHolder::invertColors(AlgoData data){
    CPLErr error;
    int i, size = data.tileXa*data.tileYa;
    float *buffer = (float *) CPLMalloc(sizeof(float)*size);

    this->controller->getInputReadMutex()->lock();
    error = data.inputBand->RasterIO( GF_Read, data.tileXo, data.tileYo, data.tileXa, data.tileYa,
                    buffer, data.tileXa, data.tileYa, GDT_Float32,
                    0, 0 );
    this->controller->getInputReadMutex()->unlock();
    if(error){
        CPLFree(buffer);
        return error;
    }
    for(i=0; i<size; i++){
        buffer[i] = data.max - buffer[i];
    }
    this->controller->getOutputWriteMutex()->lock();
    error = data.outputBand->RasterIO( GF_Write, data.tileXo, data.tileYo, data.tileXa, data.tileYa,
                    buffer, data.tileXa, data.tileYa, GDT_Float32,
                    0, 0 );
    this->controller->getOutputWriteMutex()->unlock();
    CPLFree(buffer);
    return error;
}
