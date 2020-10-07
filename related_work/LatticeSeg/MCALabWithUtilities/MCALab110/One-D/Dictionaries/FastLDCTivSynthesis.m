function sig = FastLDCTivSynthesis(ldct,bellname,w,par3)
% FastLDCTivSynthesis -- Synthesize signal from local DCT iv coefficients (orthogonal fixed folding)
%  Usage
%    sig = FastLDCTivSynthesis(ldct,bellname,w)
%  Inputs
%    ldct       local DCT iv coefficients (structure array)
%    w		width of window
%    bell       name of bell to use, defaults to 'Sine'
%  Outputs
%    sig        signal whose orthonormal local DCT iv coeff's are ldct
%
%  See Also
%   FastLDCTivAnalysis, CPAnalysis, FCPSynthesis, fold, unfold, dct_iv, packet
%
	[n,J] = dyadlength(ldct(2).coeff);
	d = floor(log2(n/w));
%
% Create Bell
%
	if nargin < 3,
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
			   [xc,xl,xr] = unfold(y(:)',bp,bm);
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
    
    
