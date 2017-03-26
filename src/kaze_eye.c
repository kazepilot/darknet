#include "kaze_eye.h"

void kaze_detect(image im, char* results)
{
    set_batch_network(&eye.net, 1);
    srand(2222222);
    int i, j;

    image sized = resize_image(im, eye.net.w, eye.net.h);
    float *X = sized.data;

    box *boxes = calloc(eye.l.w * eye.l.h * eye.l.n, sizeof(box));
    float **probs = calloc(eye.l.w * eye.l.h * eye.l.n, sizeof(float *));
    for(j = 0; j < eye.l.w * eye.l.h * eye.l.n; ++j) probs[j] = calloc(eye.l.classes + 1, sizeof(float *));

    network_predict(eye.net, X);

    get_region_boxes(eye.l, 1, 1, eye.thresh, probs, boxes, 0, 0, eye.hier_thresh);
    if (eye.l.softmax_tree && eye.nms) do_nms_obj(boxes, probs, eye.l.w * eye.l.h * eye.l.n, eye.l.classes, eye.nms);
    else if (eye.nms) do_nms_sort(boxes, probs, eye.l.w * eye.l.h * eye.l.n, eye.l.classes, eye.nms);
    int num = eye.l.w * eye.l.h * eye.l.n;

    for(i = 0; i < num; ++i){
        int class = max_index(probs[i], eye.l.classes);
        float prob = probs[i][class];
        if(prob > eye.thresh){
            box b = boxes[i];

            float left  = (b.x - b.w / 2.);
            float right = (b.x + b.w / 2.);
            float top   = (b.y - b.h / 2.);
            float bot   = (b.y + b.h / 2.);

            if(left < 0.0) left = 0.0;
            if(right > eye.l.w-1) right = 1.0;
            if(top < 0.0) top = 0.0;
            if(bot > 1.0) bot = 1.0;

            char temp[50];
			char *temp_ptr = temp;
			sprintf(temp_ptr, "%s %.2f %.2f %.2f %.2f %f\n", eye.names[class], left, top, right, bot, prob);
            strcat(results, temp_ptr);
        }
    }
    free(boxes);
    free_ptrs((void **)probs, eye.l.w * eye.l.h * eye.l.n);
}

void kaze_predict(int h, int w, int c, int step, unsigned char *data, char *results)
{
	image im = char_ptr_image(data, h, w, c, step);
	kaze_detect(im, results);
    free_image(im);
}

void kaze_init(const char *datacfg, const char *cfgfile, const char *weightfile, float thresh, float hier_thresh, float nms)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    eye.names = get_labels(name_list);

    eye.net = parse_network_cfg(cfgfile);
    load_weights(&eye.net, weightfile);
    set_batch_network(&eye.net, 1);

    eye.l = eye.net.layers[eye.net.n - 1];
    eye.thresh = thresh; // 0.2
    eye.hier_thresh = hier_thresh; // 0.5
    eye.nms = nms; //0.4
}