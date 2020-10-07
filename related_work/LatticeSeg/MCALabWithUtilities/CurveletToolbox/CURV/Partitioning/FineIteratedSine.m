function [ix, w] = FineIteratedSine(J);

dyadic_point = 2^J;

ix = [1:(2*dyadic_point)] - 1;
w  =  IteratedSine(3/2*(ix - dyadic_point)/dyadic_point);

ix = ix+1;




