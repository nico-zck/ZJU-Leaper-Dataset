function img = FastLDCT2ivSynthesis(coef,bellname,w,lfign)
% FastLDCT2Synthesis -- 2D Local inverse DCT iv transform
%  Usage
%    img = FastLDCT2Synthesis(ldct,w) 
%  Inputs
%    coef   	2D Local DCT structure array
%    w        	width of window
%    bellname 	name of bell to use, defaults to 'Sine'
%    lfign      ignore lfign% of the low-frequencies 
%		This may be useful for texture+cartoon separation.
%  Outputs
%    img	    2D reconstructed n by n image
%  Description
%    The matrix img contains image reconstructed from the CP orthobasis coeffs at depth n/w.
% See Also
%   FastLDCT2ivAnalysis, FastLDCTivSynthesis
%

	if nargin < 3 | bellname==0,
	  bellname = 'Sine';
	end
	
	[n,J] = quadlength(coef(2).coeff);
	
	if coef(2).winwidth ~= w
		error('Window width is different from given argument.');
		return;
	end
	

	img = zeros(n,n);
%
	if lfign,
	  nbox = floor(n/w);
	  for boxcnt1=0:nbox-1
	    for boxcnt2=0:nbox-1
		coef(2).coeff(boxcnt1*w+1:boxcnt1*w+1+floor(w*lfign),boxcnt2*w+1:boxcnt2*w+1+floor(w*lfign)) = 0;
	    end
	  end
	end

	for ncol=1:n
	       ldct = [struct('winwidth', w, 'coeff', 0) ...
		       struct('winwidth', w, 'coeff', coef(2).coeff(:,ncol))];
	       x = FastLDCTivSynthesis(ldct,bellname,w);
	       img(:,ncol) = x;
	   end
	   
	for nrow=1:n
	       ldct = [struct('winwidth', w, 'coeff', 0) ...
		       struct('winwidth', w, 'coeff', img(nrow,:))];
	       x = FastLDCTivSynthesis(ldct,bellname,w);
	       img(nrow,:) = x';
	end


    

    
