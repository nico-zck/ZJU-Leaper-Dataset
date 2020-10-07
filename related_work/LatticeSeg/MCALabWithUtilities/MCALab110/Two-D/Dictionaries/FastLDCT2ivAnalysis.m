function coef = FastLDCT2ivAnalysis(img,bellname,w,lfign)
% FastLDCT2ivAnalysis -- Analyze image into 2-d cosine packet coefficients at a given depth (window width)
%  Usage
%    coef = FastLDCT2ivAnalysis(img,w)
%  Inputs
%    img      	2-d image to be transformed into basis
%    w        	width of window
%    bellname 	name of bell to use, defaults to 'Sine'
%    lfign      ignore lfign % of the low-frequencies (active during synthesis) 
%		This may be useful for texture+cartoon separation.
%  Outputs
%    coef     	2-d Local DCT iv coeffts
%
%  Description
%    Once a cosine packet basis depth has been selected (at a given window width), 
%    this function may be used to expand a given
%    image in that orthobasis.
% See Also
%   FastLDCT2ivSynthesis, FastLDCTivAnalysis
%

   	if nargin < 3 | bellname==0,
	  bellname = 'Sine';
	end
    
	[n,J] = quadlength(img);
	
	d = floor(log2(n/w));

%
% CP image at depth d
%
	coef = [struct('winwidth', w, 'coeff', []) ...
		    struct('winwidth', w, 'coeff', zeros(n,n))];
%
       for nrow=1:n
           ldct = FastLDCTivAnalysis(img(nrow,:),bellname,w);
           coef(2).coeff(nrow,:) = ldct(2).coeff;
       end
       
       for ncol=1:n
           ldct = FastLDCTivAnalysis(coef(2).coeff(:,ncol),bellname,w);
           coef(2).coeff(:,ncol) = ldct(2).coeff;
       end
       



    
