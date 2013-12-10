#ifndef _SVM_SCALE_H
#define _SVM_SCALE_H

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <iostream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

void exit_with_help();
void output_target(double value);
void output(int index, double value);
char* readline(FILE *input);

std::vector<float> scale_main(int argc,char **argv, char* fscaled_name) ;

#endif
