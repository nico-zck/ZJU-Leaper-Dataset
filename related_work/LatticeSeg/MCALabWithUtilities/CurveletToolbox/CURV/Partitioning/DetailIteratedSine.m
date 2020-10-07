function [ix, w] = DetailIteratedSine(dyadic_points);


eps    = floor(dyadic_points(1)/3);
epsp   = dyadic_points(1) - eps - 1;

ix  = [dyadic_points(1)-eps+1 :  dyadic_points(2)+epsp+1 ] - 1;

wl  =  IteratedSine(3*(ix - dyadic_points(1))/dyadic_points(2));
wr =   IteratedSine(3/2 * (dyadic_points(2) - ix)/dyadic_points(2));

w   =  wl.*wr;
ix = ix+1;










