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

uint64_t compress(uint8_t* input, uint64_t input_len, uint8_t* dst, uint8_t* flags);
uint64_t decompress(uint8_t* input, uint8_t* flags, uint64_t input_len, uint8_t* dst);

/*
void char_dump_bin(unsigned char c);
void human_readable_compression(unsigned char *comp, uint64_t len);
*/
