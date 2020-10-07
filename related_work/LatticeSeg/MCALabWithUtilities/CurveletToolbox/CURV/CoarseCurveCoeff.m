function C = CoarseCurveCoeff(X);
% CoarseCurveCoeff: Returns the tight frame coefficients 
% of the coarsest scale;
%  Usage:
%      C = CoarseCurveCoeff(X);
%  Inputs:
%    X      m by m matrix, m = 2*2^L. Typically, the input is
%           obtained after multiplying the Fourier transform of an object
%           with a low frequency Meyer window   
%  Outputs:
%    C      m by m matrix  
%  Description:
%       This gives tight frame coefficients of the coarsest scale;
%       these are "Father wavelet Coeffficients" associated
%       to inverse FFT2 of the frequency domain components
%	associated to a 2^L * 2^L square.
% See Also
%   Curvelet02Xform, InvCoarseCurveCoeff

	 C = ifft2_mid0(X)*sqrt(prod(size(X)));
     