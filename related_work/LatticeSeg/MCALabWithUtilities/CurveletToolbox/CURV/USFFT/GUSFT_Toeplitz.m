function y = GUSFT_Toeplitz(x,lambda);
% GUSFT_Toeplitz: Gram matrix associated with the USFT
%  Usage:
%     y = GUSFT_Toeplitz(x,lambda);
%  Inputs:
%    x       vector of length n
%    lambda  Fourier multiplier of length 2n -1 
%  Outputs:
%    y       vector of length n 
%  Description:
%    Performs  A'A where A is USFT. A'A has a Toeplitz structure. 
%    A Toeplitz matrix is embedded in a larger circulant matrix. 
%    A circulant matrix is diagonal in a Fourier basis and 
%    lambda are the coefficients on the diagonal (i.e. lambda
%    specifies the unequispaced grid). 
%  See Also
%    USFT_simple, USFFT, MakeFourierDiagonal

        n = length(x);
	extended.x = [x;zeros(n-1,1)];
        y = fft(ifft(extended.x) .* lambda);
	y = y(1:n);
	
 
