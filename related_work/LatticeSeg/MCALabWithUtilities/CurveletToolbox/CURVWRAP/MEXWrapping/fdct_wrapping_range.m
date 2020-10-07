function [c,nall] = fdct_wrapping_range(n,nbscales)
% fdct_wrapping_range - Return a cell array of structure containing the number 
%			of angles and wedge size at each scale and orientation.
%
% Inputs
%     n         image (matrix) size
%     nbscales  Coarsest decomposition scale (default: floor(log2(min(m,n)))-3)
%
% Output
%     c         Size properties
%
% See also fdct_wrapping.m


if(~exist('nbscales'))
	nbscales = nextpow2(n);
end

nbangles_coarse = 16;


N1 = n;
N2 = n;

XL1 = 4.0*N1/3.0;  
XL2 = 4.0*N2/3.0;

nbangles = nbangles_coarse * 2.^(ceil([nbscales-2:-1:0]/2));

nd = nbangles / 4;

XS1 = 2*floor(XL1./2.^[1:nbscales-1])+1;
XS2 = 2*floor(XL2./2.^[1:nbscales-1])+1;

XL1 = XL1./2.^([0:nbscales-2]);
XL2 = XL2./2.^([0:nbscales-2]);

XW1 = XL1./nd; 
XW2 = XL2./nd;

XS1 = 2*floor(XL1/2)+1;	XS2 = 2*floor(XL2/2)+1;
XF1 = floor(XL1/2);	XF2 = floor(XL2/2); 
XR1 = XL1/2;		XR2 = XL2/2;

w = 0;
xs = XR1/4 - (XW1/2)/4;
xe = XR1;
ys = -XR2 + (w-0.5)*XW2;
ye = -XR2 + (w+1.5)*XW2;
xn = ceil(xe-xs);
yn = ceil(ye-ys);

i = find(~mod(xn,2)); 
xn(i) = xn(i) + 1;
i = find(~mod(yn,2)); 
yn(i) = yn(i) + 1;

XS1 = 2*floor(4.0*N1/3.0/2.^nbscales)+1;
XS2 = 2*floor(4.0*N1/3.0/2.^nbscales)+1;

nall = sum(nbangles.*xn.*yn) + XS1*XS2;

c{1} = struct('nbangles',1,'sw',[XS1 XS2]);
for j=2:nbscales
 m = xn(j-1);
 n = yn(j-1);
 c{nbscales-j+2} = struct('nbangles',nbangles(j-1),'sw',[repmat([m n],nbangles(j-1)/4,1); ...
 					      repmat([n m],nbangles(j-1)/4,1); ...
					      repmat([m n],nbangles(j-1)/4,1); ...
					      repmat([n m],nbangles(j-1)/4,1)]);
end
 

