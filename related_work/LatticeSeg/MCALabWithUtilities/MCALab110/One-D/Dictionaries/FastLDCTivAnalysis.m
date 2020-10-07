function ldct = FastLDCTivAnalysis(x,bellname,w,par3)
% FastLDCTivAnalysis -- Local DCT iv transform (orthogonal fixed folding)
%  Usage
%    ldct = FastLDCTivAnalysis(x,bell,w) 
%  Inputs
%    x        1-d signal:  length(x)=2^J
%    w        width of window
%    bell     name of bell to use, defaults to 'Sine'
%  Outputs
%    ldct     1-d Local DCT iv coefficients (structure array)
%  Description
%    The vector ldct contains coefficients of the Local DCT Decomposition.
% See Also
%   FastLDCTivSynthesis, CPAnalysis, FCPSynthesis, fold, unfold, dct_iv, packet
%

	if nargin < 3,
	  bellname = 'Sine';
	end
	[n,J] = dyadlength(x);
	
	d = floor(log2(n/w));
%
% taper window
%
	m = n / 2^d /2;
	[bp,bm] = MakeONBell(bellname,m);
%
% packet table
%
	n  = length(x);
	ldct = [struct('winwidth', w, 'coeff', zeros(floor(n/w),1)) ...
		    struct('winwidth', w, 'coeff', zeros(n,1))];
	x  = ShapeAsRow(x);
%
	   nbox = 2^d;
	   for b=0:(nbox-1)
		   if(b == 0) ,                         % gather packet and
			   xc = x(packet(d,b,n));           % left, right neighbors
			   xl = edgefold('left',xc,bp,bm);  % taking care of edge effects
		   else
			   xl = xc;
			   xc = xr;          
		   end
		   if (b+1 < nbox)
			   xr = x(packet(d,b+1,n));
		   else
			   xr = edgefold('right',xc,bp,bm);
		   end
		   y = fold(xc,xl,xr,bp,bm);    % folding projection
		   c = dct_iv(y);               % DCT-IV
		   ldct(2).coeff(packet(d,b,n)) = c';  % store
	   end


    
