# LZSS Compression for GPU
## LZSS Algorithm
The LZSS algorithm is a compression algorithm that takes advantage of
repeated phrases in some text.
That is, when a word is repeated within a small frame of one another,
the LZSS algorithm replaces the second occurrence of that word with a
reference to the first word.
For example, we have the phrase:
    yiwen song is a yiwen song.
We can compress to:
    yiwen song is a (16,10).

## Small Implementation details
There are quite a few ways that one can implement LZSS such that it can
compress and can be decompressed with relative ease.
I have chosen the following designs:

* Before each non-repeating character, we insert a 0 bit.
* A repeat is denoted by a pair of numbers, offset and length.
* The offset is a negative number indicating the offset 
of the occurrence the phrase.
* The length is how long the last occurrence of the phrase is
* These are stored as 16-bit integers, the former signed and
the latter unsigned.
* We use a maximum of 32 KB window--that is, the maximum number
of characters that we perform the backwards search is 32 KB 
before the current location.


## Serial Performance

## Serial Compression Rate

## GPU Performance

## GPU Compression Rate

## Conclusion
