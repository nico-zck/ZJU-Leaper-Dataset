function x = FastRealFourierSynthesis(coeff,w,lfign)
% FastRealFourierSynthesis -- 1D signal synthesis from Real Fourier coefficients
%  Usage
%    x = FastRealFourierSynthesis(coeff,w,lfign)
%  Inputs
%    coeff    	Real Fourier coefficients
%    w		window width
%    lfign	low-frequency components to ignore 
%  Outputs
%    x        	1-d signal:  length(x)=2^J
%  Description
%    c contains coefficients of the Real Fourier Decomposition.
% See Also
%   FastRealFourierAnalysis, fft, ifft
%	

    c = coeff(2).coeff;
    n = coeff(1).length;
    d = floor(n/w);
    
    x = zeros(n,1); 
    
    if lfign,
      for p=0:d-1
    	c(p*w+1:p*w+1+floor(w*lfign),:) = 0;
      end
    end
	
    for p=0:d-1
     xft = (c(p*w+1:(p+1)*w,1) + i*c(p*w+1:(p+1)*w,2));
     x(p*w+1:(p+1)*w) = sqrt(w/2)*real(ifft(xft(:)));
    end
    
