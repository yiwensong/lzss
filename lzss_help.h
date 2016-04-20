/* Header file to include for all the LZSS functionality */
#pragma once

#include <stdint.h>

#define LARGE (1024 * 1024)

typedef struct match match_t;
struct match
{
  int16_t  d;
  uint16_t l;
};

uint64_t compress(char* input, uint64_t input_len, char* dst);
uint64_t decompress(char* input, uint64_t input_len, char* dst);




void char_dump_bin(unsigned char c);
void human_readable_compression(unsigned char *comp, uint64_t len);
