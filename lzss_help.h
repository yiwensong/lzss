/* Header file to include for all the LZSS functionality */
#pragma once

#include <stdint.h>

typedef struct match match_t;
struct match
{
  int16_t  d;
  uint16_t l;
};

size_t compress(char* input, size_t input_len, char* dst);
size_t decompress(char* input, size_t input_len, char* dst);




void char_dump_bin(unsigned char c);
