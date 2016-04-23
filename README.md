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

* There is a separate flag array that is before all the data because 
memory alignment is desirable for those writing code.
* A repeat is denoted by a pair of numbers, offset and length.
* The offset is a positive number indicating how many bytes to back.
* The length is how long the last occurrence of the phrase is.
* The offset is stored as a uint16\_t, and the length is stored
as a uint8\_t.
* These numbers may change as we go to optimize for compression
ratio for various forms of documents.
* We use a maximum of 64 KB window--that is, the maximum number
of characters that we perform the backwards search is 64 KB 
before the current location.
* For file creation, we also write the size of the file to facilitate
memory allocation on the decompression.

## Serial Performance

## Serial Compression Rate

## GPU Performance

## GPU Compression Rate

## Conclusion

---
Please also visit my personal site [here](http://defdonthire.me).
