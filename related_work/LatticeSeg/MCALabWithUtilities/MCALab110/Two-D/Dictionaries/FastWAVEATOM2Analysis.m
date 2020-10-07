function wa = FastWAVEATOM2Analysis(img,pat,pars2,pars3)
% FastWAVEATOM2Analysis -- 2D Wave Atom transform (frame with redundancy 2, symmetric version)
%  Usage
%    wa = FastWAVEATOM2Analysis(img,pat)
%  Inputs
%    img	n by n image, n = 2^J 
%    pat:  	type of frequency partition which satsifies parabolic scaling relationship equal to 'p' or 'q'
%  Outputs
%    wa    	Wave Atom coefficients, a structure array (atoms are normalized to unit l_2 norm)
%  Description
%    The wa contains coefficients of the Wave Atom Decomposition.
% See Also
%   FastWAVEATOM2Synthesis
%

	[n,J] = quadlength(img);
	
	wa = [struct('pat', pat, 'coeff', 1) struct('pat', pat, 'coeff', zeros(2*n,n))];
%
	
	wa(2).coeff = sqrt(2)/n*real(fatfm2sym(img,pat));


    
