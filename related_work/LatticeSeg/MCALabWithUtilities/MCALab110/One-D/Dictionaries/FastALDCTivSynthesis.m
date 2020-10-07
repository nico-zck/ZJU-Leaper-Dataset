function sig = FastALDCTivSynthesis(ldct,bellname,D,par3)
% FastALDCTivSynthesis -- Synthesize signal from Adpative local DCT iv coefficients (orthogonal fixed folding)
%  Usage
%    sig = FastALDCTivSynthesis(ldct,bellname,D)
%  Inputs
%    ldct       local DCT coefficients (structure array)
%    n		Signal length
%    w		width of window
%    bell     name of bell to use, defaults to 'Sine'
%  Outputs
%    sig      signal whose orthonormal local DCT coeff's are ldct
%
%  See Also
%   FastALDCTivAnalysis, FastLDCTivAnalysis, FastLDCTivSynthesis, CPAnalysis, FCPSynthesis, fold, unfold, dct_iv, packet
%
	[n,J] = dyadlength(ldct(2).coeff);
	w = ldct(2).winwidth;
	d = floor(log2(n/w));
%
% Create Bell
%
	if nargin < 2,
	  bellname = 'Sine';
	end
	m = n / 2^d /2;
	[bp,bm] = MakeONBell(bellname,m);
%
%
%
		x = zeros(1,n);
		for b=0:(2^d-1),
			   c = ldct(2).coeff(packet(d,b,n));
			   y = dct_iv(c);
			   [xc,xl,xr] = unfold(y,bp,bm);
			   x(packet(d,b,n)) = x(packet(d,b,n)) + xc;
			   if b>0,
				   x(packet(d,b-1,n)) = x(packet(d,b-1,n)) + xl;
			   else
			       x(packet(d,0,n))   = x(packet(d,0,n)) + edgeunfold('left',xc,bp,bm);
			   end
			   if b < 2^d-1,
				   x(packet(d,b+1,n)) = x(packet(d,b+1,n)) + xr;
			   else         
			       x(packet(d,b,n))   = x(packet(d,b,n)) + edgeunfold('right',xc,bp,bm);
			   end
		 end
		 sig = ShapeAsRow(x)';
    
    
