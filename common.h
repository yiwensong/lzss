#pragma once

#include <stdint.h>

/* TIMING */
double read_timer();

/* argument processing */
int find_option( int argc, char **argv,const char *option );
int read_int( int argc, char **argv, const char *option, int default_value);
char *read_string( int argc, char **argv, const char *option, char *default_value);

/* IO */
uint64_t file_size(const char *filename);
