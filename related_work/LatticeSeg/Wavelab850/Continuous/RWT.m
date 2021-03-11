function rwt = RWT(x,nvoice,wavelet,oct,scale)
% RWT -- Real Wavelet Transform
%  Usage
%    rwt = RWT(x,nvoice,wavelet)
%  Inputs
%    x        signal, dyadic length n=2^J, real-valued
%    nvoice   number of voices/octave
%    wavelet  string 'Gauss', 'DerGauss','Sombrero', 'Morlet'
%  Outputs
%    rwt      matrix n by nscale where
%             nscale = nvoice .* noctave
%
%  Description
%     see sections 4.3.1 and 4.3.3 of Mallat's book  
%
	if nargin<4,
		oct = 2;
		scale = 4;
	end	
% preparation
	x = ShapeAsRow(x);
	n = length(x);
	xhat = fft(x);
	xi   = [ (0: (n/2)) (((-n/2)+1):-1) ] .* (2*pi/n);
	
% root
	omega0 = 5;
	
%	noctave = floor(log2(n))-2;
%	noctave = floor(log2(n))-1;
	noctave = floor(log2(n))-oct;
	nscale  = nvoice .* noctave;
	
	rwt = zeros(n,nscale);

	kscale  = 1;
%	scale   = 4;
%	scale = 16;

	for jo = 1:noctave,
	    for jv = 1:nvoice,
		   qscale = scale .* (2^(jv/nvoice));
		   omega =  n .* xi ./ qscale ;
		   if strcmp(wavelet,'Gauss'),
				window = exp(-omega.^2 ./2);
		elseif strcmp(wavelet,'DerGauss'),
                                window = i.*omega.*exp(-omega.^2 ./2);
		   elseif strcmp(wavelet,'Sombrero'),
				window = (omega.^2) .* exp(-omega.^2 ./2);
		   elseif strcmp(wavelet,'Morlet'),
				window = exp(-(omega - omega0).^2 ./2) - exp(-(omega.^2 + omega0.^2)/2);
		   end
%(  Renormalisation comme dans le bouquin )
		   window = window ./ sqrt(qscale);
		   what = window .* xhat;
		   w    = ifft(what);
		   rwt(1:n,kscale) = real(w)';
		   kscale = kscale+1;
		end
		scale  = scale .*2;
    end 
% kscale = 1 -> low frequencies
% the matrix rwt is ordered from low to high frequencies 

% Written by Maureen Clerc and Jerome Kalifa, 1997
% clerc@cmapx.polytechnique.fr, kalifa@cmapx.polytechnique.fr

   
    
 
 
%
%  Part of Wavelab Version 850
%  Built Tue Jan  3 13:20:39 EST 2006
%  This is Copyrighted Material
%  For Copying permissions see COPYING.m
%  Comments? e-mail wavelab@stat.stanford.edu 
