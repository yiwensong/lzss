/* Header file to include for all the LZSS functionality */
#pragma once

#include <stdint.h>

#define LARGE (1024 * 1024)

#define BITS_TO_CHARS(b) (((b)+8-1)/8)
#define FLAG_BYTES(compressed) (BITS_TO_CHARS(((compressed)->flag_bits)))

typedef struct match_expanded match_expanded_t;
struct match_expanded
{
  uint16_t d;
  uint16_t l;
};

typedef struct match match_t;
struct match
{
  uint16_t dl;
};

typedef struct compressed compressed_t;
struct compressed
{
  uint64_t file_len;
  uint64_t content_len;
  uint64_t flag_bits;
  uint8_t content[0];
};

typedef struct decomp decomp_t;
struct decomp
{
  uint64_t content_len;
  uint8_t content[0];
};

typedef struct comp_size comp_size_t;
struct comp_size
{
  uint64_t b;
  uint64_t w;
};

comp_size_t compress(uint8_t* input, uint64_t input_len, uint8_t* dst, uint8_t* flags);
uint64_t decompress(uint8_t* input, uint8_t* flags, uint64_t input_len, uint8_t* dst);

/* The following two methods requires a call to free on the returned pointer when finished */
compressed_t *lzss_compress(decomp_t *decomp);
decomp_t *lzss_decomp(compressed_t *comp);

/*
void char_dump_bin(unsigned char c);
void human_readable_compression(unsigned char *comp, uint64_t len);
*/
