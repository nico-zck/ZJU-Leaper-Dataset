function [t,w] = NewDetailMeyerWindow(j);

lo = floor(2*2^j/3);
t = 0:(2^(j+1)-1);
t = lo + t;

w = IteratedSine(3/2*2^(-j)*(t-2^j)).*IteratedSine(3/4*2^(-j)*(2^(j+1)-t));


