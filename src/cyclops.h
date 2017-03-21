#ifndef CYCLOPS_H
#define YCLOPS_H


#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif


typedef struct {
    network net;
    layer l;
    float thresh;
    float hier_thresh;
    float nms;

} Cyclops;

Cyclops cyclops;

void train(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear);
void init(char *datacfg, char *cfgfile, char *weightfile, float thresh, float hier_thresh, float nms);
void detect(float *img_data);
void cyclops_destroy();


#endif