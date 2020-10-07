function [ix, w] = CoarseIteratedSine(L);


dyadic_point = 2^L;
ix = [1:(2*dyadic_point)] -1; 

w = IteratedSine(3/2 * (dyadic_point - ix)/dyadic_point);
ix = ix+1;











