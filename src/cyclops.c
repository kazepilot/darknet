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

void train(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network *nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = parse_network_cfg(cfgfile);
        if(weightfile){
            load_weights(&nets[i], weightfile);
        }
        if(clear) *nets[i].seen = 0;
        nets[i].learning_rate *= ngpus;
    }
    srand(time(0));
    network net = nets[0];

    int imgs = net.batch * net.subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    data train, buffer;

    layer l = net.layers[net.n - 1];

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    args.threads = 8;

    args.angle = net.angle;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;

    pthread_t load_thread = load_data(args);
    clock_t time;
    int count = 0;
    while(get_current_batch(net) < net.max_batches){
        if(l.random && count++%10 == 0){
            printf("Resizing\n");
            int dim = (rand() % 10 + 10) * 32;
            if (get_current_batch(net)+200 > net.max_batches) dim = 608;
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            for(i = 0; i < ngpus; ++i){
                resize_network(nets + i, dim, dim);
            }
            net = nets[0];
        }
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);
        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}


void detect(float *img_data)
{
    set_batch_network(&cyclops.net, 1);
    srand(2222222);
    char buff[256];
    char *input = buff;
    int j;

    box *boxes = calloc(cyclops.l.w * cyclops.l.h * cyclops.l.n, sizeof(box));
    float **probs = calloc(cyclops.l.w * cyclops.l.h * cyclops.l.n, sizeof(float *));
    for(j = 0; j < cyclops.l.w * cyclops.l.h * cyclops.l.n; ++j) probs[j] = calloc(cyclops.l.classes + 1, sizeof(float *));

    network_predict(cyclops.net, img_data);
    get_region_boxes(cyclops.l, 1, 1, cyclops.thresh, probs, boxes, 0, 0, cyclops.hier_thresh);
    if (cyclops.l.softmax_tree && cyclops.nms) do_nms_obj(boxes, probs, cyclops.l.w * cyclops.l.h * cyclops.l.n, cyclops.l.classes, cyclops.nms);
    else if (cyclops.nms) do_nms_sort(boxes, probs, cyclops.l.w * cyclops.l.h * cyclops.l.n, cyclops.l.classes, cyclops.nms);
    //draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);

    free(boxes);
    free_ptrs((void **)probs, cyclops.l.w * cyclops.l.h * cyclops.l.n);
}

void init(char *datacfg, char *cfgfile, char *weightfile, float thresh, float hier_thresh, float nms)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    cyclops.net = parse_network_cfg(cfgfile);
    load_weights(&cyclops.net, weightfile);

    cyclops.l = cyclops.net.layers[cyclops.net.n-1];
    cyclops.thresh = thresh; // 0.2
    cyclops.hier_thresh = hier_thresh; // 0.5
    cyclops.nms = nms; //0.4
}

void destroy()
{

}