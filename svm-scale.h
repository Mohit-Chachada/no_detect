#ifndef _SVM_SCALE_H
#define _SVM_SCALE_H

void exit_with_help();
void output_target(double value);
void output(int index, double value);
char* readline(FILE *input);

void scale_main(int argc,char **argv, char* fscaled_name) ;

#endif
