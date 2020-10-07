function c = FastRealFourierAnalysis(x,w,lfign,pars3)
% FastRealFourierAnalysis -- Real Fourier transform for 1D signals
%  Usage
%    c = FastRealFourierAnalysis(x,w,lfign) 
%  Inputs
%    x        	1-d signal:  length(x)=2^J
%    w		window width
%    lfign	low-frequency components to ignore (active during synthesis)
%  Outputs
%    c    	Real Fourier coefficients
%  Description
%    c contains coefficients of the Real Fourier Decomposition.
% See Also
%   FastRealFourierSynthesis, fft, ifft
%		

    
	[n,J] = dyadlength(x);
	
	d = floor(n/w);
	
	c = [struct('coeff',zeros(d,1),'width',w,'length',n) ...
	     struct('coeff',zeros(n,2),'width',w,'length',n)];


	for p=0:d-1	
	  xft    = fft(x(p*w+1:(p+1)*w))*sqrt(2/w);
	  c(1).coeff(p+1) = xft(1);
	  c(2).coeff(p*w+1:(p+1)*w,1) = real(xft(:));
	  c(2).coeff(p*w+1:(p+1)*w,2) = imag(xft(:));
	end
