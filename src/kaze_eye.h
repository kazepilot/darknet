#ifndef KAZEEYE_H
#define KAZEEYE_H

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
    char **names;
    box *boxes;
    float **probs;

} KazeEye;

KazeEye eye;

#ifdef __cplusplus
extern "C" {
#endif

void kaze_init(const char *datacfg, const char *cfgfile, const char *weightfile, float thresh, float hier_thresh, float nms);
void kaze_predict(int h, int w, int c, int step, unsigned char *data, char *results);

#ifdef __cplusplus
}
#endif

void kaze_detect(image im, char* results);

#endif