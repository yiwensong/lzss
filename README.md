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
* The offset is a 11 bit integer and the length is a 4 bit integer
Together they are stored as a single uint16\_t.
* We use a maximum of 2 KB window--that is, the maximum number
of characters that we perform the backwards search is 2 KB 
before the current location.
* For file creation, we also write the size of the file to facilitate
memory allocation on the decompression.

## Serial Performance
The performance of the serial compression algorithm is compared to a 
reference implementation that was found online.
With the same window size and maximum match lengths, this implementation
of LZSS is very similar to the reference implementation.

## Serial Compression Rate
The deflation of this implementation was tested on several types of files.
The first type of file is a highly compressable file, in which most of
the file was repeated characters within the allowable window size.
This sort of file had a compression ratio up to 15.3, which equates to
the compressed file having 7% of the original file size.

## GPU Performance
The GPU performance hit a peak of 5 times faster than the serial implementation.

## GPU Compression Rate
The GPU algorithm is does not change the ordering or chunking of
the serial implementation.
This means that the compression ratio of the GPU will be exactly the same
as that of the serial implementation.

## Conclusion

---
Please also visit my personal site [here](http://defdonthire.me).
