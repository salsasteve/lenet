/*
 * main.c
 * Include processing inputs, outputs and applying deep learning models
 */

#include "main.h"
// static uint16_t i = 0, k = 0, j = 0, m = 0;

#pragma PERSISTENT(LAYER_0)
static layer_t LAYER_0 = {.class=Conv2d, .activation=&fp_tanh, .kernel=&CONV2D_1_WEIGHTS_MAT, .bias=&CONV2D_1_BIASES_MAT,
                                                    .numChannels=1, .numFilters=8, .padding=Same, .stride_numCols=1, .stride_numRows=1};

// #pragma PERSISTENT(LAYER_1)
// static layer_t LAYER_1 = {.class=Maxpooling, .activation=&fp_linear, .numChannels=8, .padding=Valid, .stride_numCols=3, .stride_numRows=3, .pool_numCols=3, .pool_numRows=3};

// #pragma PERSISTENT(LAYER_2)
// static layer_t LAYER_2 = {.class=Conv2d, .activation=&fp_relu, .kernel=&MNIST_KERNEL_1_MAT, .bias=&MNIST_BIAS_1_MAT,
//                                                     .numChannels=8, .numFilters=16, .padding=Same, .stride_numCols=1, .stride_numRows=1};

// #pragma PERSISTENT(LAYER_3)
// static layer_t LAYER_3 = {.class=Maxpooling, .activation=&fp_linear, .numChannels=16, .padding=Valid, .stride_numCols=3, .stride_numRows=3, .pool_numCols=3, .pool_numRows=3};

// #pragma PERSISTENT(LAYER_4)
// static layer_t LAYER_4 = {.class=Flatten, .numChannels=16};

// #pragma PERSISTENT(LAYER_5)
// static layer_t LAYER_5 = {.class=Dense, .activation=&fp_relu, .kernel=&MNIST_KERNEL_2_MAT, .bias=&MNIST_BIAS_2_MAT};

// #pragma PERSISTENT(LAYER_6)
// static layer_t LAYER_6 = {.class=Dense, .activation=&fp_relu, .kernel=&MNIST_KERNEL_3_MAT, .bias=&MNIST_BIAS_3_MAT};

// #pragma PERSISTENT(LAYERS)
// static layer_t *LAYERS[7] = {&LAYER_0, &LAYER_1, &LAYER_2, &LAYER_3, &LAYER_4, &LAYER_5, &LAYER_6};

#pragma PERSISTENT(LAYERS)
static layer_t *LAYERS[7] = {&LAYER_0};

#pragma PERSISTENT(MODEL)
static model_t MODEL = {.numLayers=1, .layers=LAYERS, .input=&MNIST_INPUT_MAT, .output=&MNIST_OUTPUT_MAT};

void mnist_test(){
   predict(&MODEL);
}

void main(void){

    /* stop watchdog timer */
    WDTCTL = WDTPW | WDTHOLD;

    /* initialize GPIO System */
    init_gpio();

    /* initialize the clock and baudrate */
    init_clock_system();

    // evaluate(128,256);
    mnist_test();
}