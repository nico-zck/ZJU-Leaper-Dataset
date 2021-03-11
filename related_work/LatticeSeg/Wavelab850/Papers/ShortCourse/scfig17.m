% scfig17 -- Short Course 17: Noisy Differentiation by Ideal Fourier Damping
%
% This figure displays the results of an ideal Fourier Damping
% for deconvolution, in which each frequency is damped optimally
% based on its signal-to-noise ratio. For comparison, the Naive
% reconstruction based on simple differencing is also shown.
%
% Unlike the wavelet thresholding of Figure 15, ideal Fourier
% damping is not able to suppress noise while preserving resolution.
%
global Bumps
%
	sig = cumsum(Bumps) + WhiteNoise(Bumps);
	x = diff([0 sig]);
%
	z = fft(x);
	w = fft(Bumps);
	omega = 2 * pi .* (0:(length(w)-1)) ./ length(w);
	sigma2 = length(x) .* 4 .* sin(omega/2).^2;
	wate = abs(w).^2 ./ (abs(w).^2 + sigma2);
	whate = wate .* z;
	BumpsHat= real(ifft(whate));
%
% 	clf;
	s = linspace(0,1,length(x)); 
	subplot(221); plot(s,Bumps) ;    title('17 (a) Object')
	subplot(222); plot(s,sig);       title('17 (b) Noisy Cumulative')
	subplot(223); plot(s,x) ;        title('17 (c) Difference of Noisy Cumulative')
	subplot(224); plot(s,BumpsHat) ; title('17 (d) Ideal Fourier Damping')
    
    
    
 
 
%
%  Part of Wavelab Version 850
%  Built Tue Jan  3 13:20:42 EST 2006
%  This is Copyrighted Material
%  For Copying permissions see COPYING.m
%  Comments? e-mail wavelab@stat.stanford.edu 
