function imr = FastWAVEATOM2Synthesis(wa,pat,pars2,pars3)
% FastWAVEATOM2Synthesis -- (pseudo-)inverse 2D Wave Atom transform (frame with redundancy 2, symmetric version)
%  Usage
%    wa = FastWAVEATOM2Analysis(img,pat)
%  Inputs
%    wa    	Wave Atom coefficients, a structure array (atoms are normalized to unit l_2 norm)
%    pat:  	type of frequency partition which satsifies parabolic scaling relationship equal to 'p' or 'q'
%  Outputs
%    imr	n by n image, n = 2^J 
%  Description
%    imr contains reconstruction from coefficients of the Wave Atom Decomposition.
% See Also
%   FastWAVEATOM2Analysis
%

	n = size(wa(2).coeff,2);
	if size(wa(2).coeff,1)~=2*n
		disp('Improper Wave Atom array structure size.');
		return;
	end
	
%
	
	imr = real(iatfm2sym(wa(2).coeff/sqrt(n/2),pat));


    
