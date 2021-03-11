function y = Evaluate_FT(x,shift,boxlen,center,w)
% Evaluate_FT -- 1d unequispaced Fourier transform
% Usage:
%   y = Evaluate_FT(x,shift,boxlen,center,w)
% Inputs:
%   x	      vector of length n
%   shift     vector of shifts
%   boxlen    half lenght of interval around each shift
%   center    boolean variable: 0/1 = unbiased/biased FT
%   w         tapering window of size 2*boxlen
% Outputs:
%   y      vector of length: (# shifts) * (2*boxlen)
% Description:
%  Evaluates the FT at the frequency points 
%                   omega(j,k) = shift(j) + k, -boxlen <= k <boxlen
%  If center = 0, 
%
%  y = sum_{-n/2 <= t < n/2} exp(-i 2pi omega t/n) x(t), -n/2 <= t < n/2
%
%  If center = 1, 
%
%  y = sum_{0 <= t < n} exp(-i 2pi omega t/n) x(t), 0 <= t < n
%
%  Evaluate_FT uses two different methods for computing the
%  unesquispaced Fourier transform. If the number of shifts is less
%  than 8, it uses an exact method, otherwise it uses the
%  accelerated USFFT. 
%  See Also
%    Adj_Evaluate_FT, USFFT, USFT_simple

if nargin < 5,
  w = ones(1,2*boxlen);
end

if nargin < 4,
  center = 0;
end

        n = length(x);
	boxcnt = length(shift);
	
        if (boxcnt <= 16)
	  y = USFT_simple(x,shift,boxlen,center,w);
	else
	  % Make grid
	  k = (-boxlen):(boxlen -1);
	  omega = meshgrid(shift,k) + meshgrid(k,shift).';
	  omega = 2*pi*(omega(:))./n;
	  % Take USFFT and multiply by window
	  y = USFFT(x,omega,16,6,center)./sqrt(n);
	  y = repmat(w(:),boxcnt,1) .* y;
	end

	
	
	
	
	
	

	
		
	
	
	
		  		  
	
	
	
		
	
	

