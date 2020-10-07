function aldct = FastALDCTivAnalysis(x,bellname,D,par3)
% FastALDCTivAnalysis -- Adaptive Local DCT iv transform (orthogonal fixed folding)
%  Usage:
%	[aldct,bwidth] = FastALDCTivAnalysis(x,bellname,D)
%  Inputs:
%	x	the signal, a column vector
%	D	the depth of wavelet packet
%    	bell    name of bell to use, defaults to 'Sine'
%  Outputs:
%	c	the coefs, structure array with the
%               best estimated width using the l^1 cost function
%  Description
%    The vector aldct contains coefficients of the Adpative Local DCT iv Decomposition.
% See Also
%   FastALDCTivSynthesis, FastLDCTivAnalysis, FastLDCTivSynthesis, CPAnalysis, FCPSynthesis

if nargin < 3,
	  bellname = 'Sine';
end

[n,J] = dyadlength(x);

%
% packet table
%	
cp = CPAnalysis(x, D, bellname);

%
%   Find Best Basis
%
p=abs(cp)/norm(cp(:,1));
stree=sum(p);
bestd=find(stree==min(stree));
bwidth=n/2^(bestd-1);
d = floor(n/bwidth);
%
aldct = [struct('winwidth', bwidth, 'coeff', zeros(d,1)) ...
	 struct('winwidth', bwidth, 'coeff', cp(:,bestd))];
		

