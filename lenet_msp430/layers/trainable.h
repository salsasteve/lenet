/*
 * trainable.h
 */

#include "layers.h"

#ifndef LAYERS_TRAINABLE_H_
#define LAYERS_TRAINABLE_H_

matrix *gradient_descent(matrix *kernel, matrix *gradient, matrix *bias, matrix *bias_gradient, uint16_t m);

int16_t mse_quad_cost(matrix *cost, uint16_t precision);
matrix *mse_delta(matrix *delta, matrix *target, matrix *predict);
matrix *mse_kernel_gradient(matrix *gradient, matrix *delta, matrix *input, int16_t rate, uint16_t precision);
matrix *mse_bias_gradient(matrix *bias_gradient, matrix *delta, int16_t rate, uint16_t precision);
matrix *mse_back_propagation(matrix *prev_delta, matrix *kernel, matrix *next_delta, uint16_t precision);

int16_t cce_loss(matrix *predict, uint16_t target, uint16_t precision);
matrix *cce_kernel_gradient(matrix *gradient, matrix *predict, matrix *input, uint16_t target, int16_t rate, uint16_t precision);
matrix *cce_bias_gradient(matrix *bias_gradient, matrix *predict, uint16_t target, int16_t rate, uint16_t precision);

int16_t sparsemax_loss(matrix *predict, uint16_t target, int16_t e, uint16_t precision);
matrix *sparsemax_kernel_gradient(matrix *gradient, matrix *predict, matrix *input, uint16_t target, int16_t rate, uint16_t precision);
matrix *sparsemax_bias_gradient(matrix *bias_gradient, matrix *predict, uint16_t target, int16_t rate, uint16_t precision);

int16_t KL_divergence(matrix *predict, uint16_t target, uint16_t precision);
#endif /* LAYERS_TRAINABLE_H_ */
