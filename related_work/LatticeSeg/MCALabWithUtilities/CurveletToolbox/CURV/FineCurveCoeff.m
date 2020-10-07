function C = FineCurveCoeff(X);
% FineCurveCoeff: Returns the tight frame coefficients at the finest scale
%  Usage:
%    C = FineCurveCoeff(X);
%  Inputs: 
%    X   n by n matrix. Typically, the input is the windowed Fourier
%        transform of an object. The window isolates the frequencies in
%        the last subband 
%  Outputs:
%    C   n by n matrix of wavelet coefficients  
%  Description:
%       This gives tight frame coefficients of the fine scale;
%       these are "Mother wavelet Coeffficients" associated
%       to inverse FFT2 of the frequency domain components
%	associated to an n*n square in the frequency domain
%       with the interior n/2 * n/2 square deleted,
%	but with smooth windowing rather than sharp cutoff

	 C = ifft2_mid0(X)*sqrt(prod(size(X)));
     
	 